#!/usr/bin/env python3
"""Accuracy benchmark: run the whisperbox pipeline against YouTube reference transcripts.

The harness:

1. Pulls YouTube's auto-generated captions via ``yt-dlp --write-auto-sub --convert-subs vtt``.
2. Pulls the audio track via ``yt-dlp -x --audio-format wav``.
3. Runs the whisperbox pipeline on the audio.
4. Computes Word Error Rate (WER) plus jiwer's MER/WIL/WIP between our
   transcript and the YouTube reference, after lowercasing, stripping
   punctuation, and collapsing whitespace.
5. Writes a JSON report to ``benchmarks/<video-id>/<UTC-timestamp>.json``.

YouTube auto-captions are themselves ASR output and may be noisy; treat WER
against them as a *parity check*, not ground truth. For higher-fidelity
comparison, supply ``--reference path/to/hand-corrected.vtt``.

Usage::

    python -m scripts.benchmark <youtube-url> [--engine {whisper,parakeet}] [--model NAME]
    python -m scripts.benchmark <url> --reference my_truth.vtt

Requires ``yt-dlp`` (system or pip) and ``jiwer`` (see ``requirements-dev.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import jiwer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config  # noqa: E402
from src.service import TranscriptionService  # noqa: E402


_VTT_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} -->")
_VTT_TAG_RE = re.compile(r"<[^>]+>")


def strip_vtt_text(vtt_content: str) -> str:
    """Return only the spoken-word payload from a WebVTT subtitle file."""
    lines = []
    for raw in vtt_content.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "NOTE", "STYLE", "Kind:", "Language:")):
            continue
        if _VTT_TIMESTAMP_RE.match(line):
            continue
        if line.isdigit():  # cue number
            continue
        # Strip in-cue style/voice/timing tags like <c.colorFFFFFF> or <00:00:01.500>.
        cleaned = _VTT_TAG_RE.sub("", line)
        if cleaned:
            lines.append(cleaned)
    # YouTube auto-captions often repeat the same line in adjacent cues for
    # rolling-text effect. Dedupe consecutive duplicates.
    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return " ".join(deduped)


def fetch_youtube(url: str, out_dir: Path) -> Tuple[Path, Path, str, dict]:
    """Pull audio + the best available English captions for ``url``.

    Prefers manual (uploader-provided) captions over auto-generated, since
    auto-captions are themselves ASR output and noisy. Falls back to
    auto-generated if no manual track exists. Records which source was used
    in the returned metadata dict under ``ref_source``.
    """
    info = subprocess.run(
        ["yt-dlp", "--dump-json", "--skip-download", url],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(info.stdout)
    video_id = meta["id"]
    duration = meta.get("duration")
    title = meta.get("title")
    out_prefix = str(out_dir / f"{video_id}")

    # Try manual captions first; fall back to auto-generated.
    ref_source = None
    subs_path: Path | None = None
    for flag, label in (("--write-sub", "manual"), ("--write-auto-sub", "auto")):
        subprocess.run(
            [
                "yt-dlp", flag, "--sub-langs", "en",
                "--skip-download", "--convert-subs", "vtt",
                "-o", out_prefix + ".%(ext)s", url,
            ],
            check=True, capture_output=True, text=True,
        )
        candidate = Path(out_prefix + ".en.vtt")
        if candidate.exists():
            subs_path = candidate
            ref_source = f"yt-dlp {label} captions (en)"
            break

    if subs_path is None:
        raise FileNotFoundError(
            f"No English captions (manual or auto) available for {video_id}. "
            "Pick a video with English captions, or pass --reference."
        )

    subprocess.run(
        [
            "yt-dlp", "-x", "--audio-format", "wav",
            "-o", out_prefix + ".%(ext)s", url,
        ],
        check=True, capture_output=True, text=True,
    )
    audio_path = Path(out_prefix + ".wav")
    if not audio_path.exists():
        candidates = list(out_dir.glob(f"{video_id}*.wav"))
        if not candidates:
            raise FileNotFoundError(f"Audio download failed for {video_id}")
        audio_path = candidates[0]

    return audio_path, subs_path, video_id, {
        "title": title, "duration": duration, "ref_source": ref_source,
    }


def normalize_for_wer(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)  # keep apostrophes for contractions
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(reference: str, hypothesis: str) -> dict:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref or not hyp:
        return {
            "wer": None, "mer": None, "wil": None, "wip": None,
            "hits": 0, "substitutions": 0, "insertions": 0, "deletions": 0,
            "reference_word_count": len(ref.split()),
            "hypothesis_word_count": len(hyp.split()),
        }
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": out.wer,
        "mer": out.mer,
        "wil": out.wil,
        "wip": out.wip,
        "hits": out.hits,
        "substitutions": out.substitutions,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "reference_word_count": len(ref.split()),
        "hypothesis_word_count": len(hyp.split()),
    }


def run_pipeline(audio_path: Path, engine: str, model: str | None) -> Tuple[dict, float]:
    if engine:
        os.environ["TRANSCRIPTION_ENGINE"] = engine
    config_kwargs = {"include_diarization": False}
    if model:
        config_kwargs["whisper_model"] = model
    config = Config(**config_kwargs)
    service = TranscriptionService(config)
    t0 = time.time()
    result = service.transcribe_file(str(audio_path))
    elapsed = time.time() - t0
    return result, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="YouTube URL (or local file path if --reference is given)")
    parser.add_argument("--engine", default=None, choices=["whisper", "parakeet"],
                        help="ASR engine override (defaults to TRANSCRIPTION_ENGINE env or whisper)")
    parser.add_argument("--model", default=None,
                        help="Override the whisper model size (e.g. tiny, base, large-v3-turbo)")
    parser.add_argument("--reference", default=None,
                        help="Path to a reference VTT file. If supplied, the URL argument is "
                             "treated as a local audio file and yt-dlp is not called.")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep downloaded audio/subs under benchmarks/_tmp/")
    args = parser.parse_args(argv)

    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)

    if args.reference:
        audio_path = Path(args.url)
        if not audio_path.exists():
            print(f"audio file not found: {audio_path}", file=sys.stderr)
            return 2
        subs_path = Path(args.reference)
        video_id = audio_path.stem
        meta = {"title": None, "duration": None}
        print(f"[1/4] Using local audio={audio_path.name}  reference={subs_path.name}", file=sys.stderr)
    else:
        work_dir = benchmarks_dir / "_tmp"
        work_dir.mkdir(exist_ok=True)
        print(f"[1/4] Fetching reference transcript + audio via yt-dlp", file=sys.stderr)
        audio_path, subs_path, video_id, meta = fetch_youtube(args.url, work_dir)
        print(f"      audio={audio_path.name}  subs={subs_path.name}  id={video_id}", file=sys.stderr)

    print(f"[2/4] Running whisperbox pipeline (engine={args.engine or os.getenv('TRANSCRIPTION_ENGINE') or 'whisper'})", file=sys.stderr)
    result, elapsed = run_pipeline(audio_path, args.engine, args.model)
    print(f"      done in {elapsed:.1f}s  segments={len(result['segments'])}", file=sys.stderr)

    print(f"[3/4] Computing WER against reference", file=sys.stderr)
    reference_text = strip_vtt_text(subs_path.read_text(encoding="utf-8"))
    hypothesis_text = " ".join(text for _, _, text, _ in result["segments"]).strip()
    metrics = compute_metrics(reference_text, hypothesis_text)

    rtf = (elapsed / meta["duration"]) if meta.get("duration") else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "url": args.url,
        "video_id": video_id,
        "video_title": meta.get("title"),
        "video_duration_s": meta.get("duration"),
        "engine": args.engine or os.getenv("TRANSCRIPTION_ENGINE") or "whisper",
        "model": args.model or os.getenv("WHISPER_MODEL") or "default",
        "ref_source": (
            meta.get("ref_source") if not args.reference else f"file:{args.reference}"
        ) or "unknown",
        "transcribe_seconds": elapsed,
        "real_time_factor": rtf,
        "segments": len(result["segments"]),
        "metrics": metrics,
        "reference_preview": reference_text[:240],
        "hypothesis_preview": hypothesis_text[:240],
        "timestamp": timestamp,
    }

    out_path = benchmarks_dir / video_id / f"{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[4/4] WER={metrics['wer']:.3f}  MER={metrics['mer']:.3f}  WIL={metrics['wil']:.3f}", file=sys.stderr)
    print(f"      ref_words={metrics['reference_word_count']}  hyp_words={metrics['hypothesis_word_count']}", file=sys.stderr)
    print(f"      report saved to {out_path}", file=sys.stderr)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not args.keep_files and not args.reference:
        work_dir = benchmarks_dir / "_tmp"
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
