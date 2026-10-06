"""Unit tests for scripts/benchmark.py helpers and engine reporting (no network)."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_BENCHMARK_PATH = _REPO_ROOT / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_module", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)


class TestStripVttText:
    def test_drops_header_and_timestamps(self):
        vtt = (
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:00.000 --> 00:00:01.500\n"
            "Hello there.\n"
            "\n"
            "00:00:01.600 --> 00:00:03.000\n"
            "General Kenobi.\n"
        )
        assert benchmark.strip_vtt_text(vtt) == "Hello there. General Kenobi."

    def test_strips_inline_tags(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.500\n"
            "<c.colorFFFFFF>Hello</c> <00:00:00.500>there.\n"
        )
        # Words after tag removal should be ``Hello there.`` (the timing tag
        # ``<00:00:00.500>`` collapses to nothing, leaving a stray space which
        # is fine — we're stripping for WER comparison, not display.)
        out = benchmark.strip_vtt_text(vtt)
        assert "Hello" in out
        assert "there." in out
        assert "<" not in out

    def test_dedupes_consecutive_duplicate_lines(self):
        # YouTube's rolling-text auto-captions repeat the same line in
        # consecutive cues; the harness should collapse those.
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000\nrepeating\n\n"
            "00:00:01.000 --> 00:00:02.000\nrepeating\n\n"
            "00:00:02.000 --> 00:00:03.000\nrepeating once more\n"
        )
        assert benchmark.strip_vtt_text(vtt) == "repeating repeating once more"

    def test_handles_cue_numbers(self):
        vtt = (
            "WEBVTT\n\n"
            "1\n00:00:00.000 --> 00:00:01.000\nFirst.\n\n"
            "2\n00:00:01.000 --> 00:00:02.000\nSecond.\n"
        )
        assert benchmark.strip_vtt_text(vtt) == "First. Second."

    def test_skips_styles_and_notes(self):
        vtt = (
            "WEBVTT\n"
            "NOTE this is a note\n"
            "STYLE\n"
            "::cue { background: black; }\n"
            "\n"
            "00:00:00.000 --> 00:00:01.000\nReal text.\n"
        )
        assert "note" not in benchmark.strip_vtt_text(vtt).lower()


class TestNormalizeForWer:
    def test_lowercases(self):
        assert benchmark.normalize_for_wer("Hello WORLD") == "hello world"

    def test_strips_punctuation(self):
        assert benchmark.normalize_for_wer("Hello, world!") == "hello world"

    def test_keeps_apostrophes(self):
        # Contractions should survive so "don't" ≠ "do n't" doesn't blow up WER.
        assert benchmark.normalize_for_wer("Don't stop") == "don't stop"

    def test_collapses_whitespace(self):
        assert benchmark.normalize_for_wer("hello   \t  world\n\n") == "hello world"


class TestComputeMetrics:
    def test_perfect_match_is_zero_wer(self):
        out = benchmark.compute_metrics("hello world", "hello world")
        assert out["wer"] == 0.0
        assert out["substitutions"] == 0
        assert out["insertions"] == 0
        assert out["deletions"] == 0

    def test_one_substitution(self):
        out = benchmark.compute_metrics("hello world", "hello there")
        assert out["wer"] == pytest.approx(0.5)
        assert out["substitutions"] == 1

    def test_empty_inputs_return_none(self):
        out = benchmark.compute_metrics("", "anything")
        assert out["wer"] is None

    def test_normalizes_punctuation_before_compare(self):
        out = benchmark.compute_metrics("Hello, world!", "hello world")
        assert out["wer"] == 0.0

    def test_metric_keys_present(self):
        out = benchmark.compute_metrics("a b c d", "a x c y")
        for key in ("wer", "mer", "wil", "wip", "hits", "substitutions",
                    "insertions", "deletions", "reference_word_count",
                    "hypothesis_word_count"):
            assert key in out


class TestReportedEngine:
    """The engine label must name the engine that actually ran, not a guess
    from argv/env — otherwise WER comparisons between engines are corrupted."""

    @pytest.fixture
    def run_benchmark(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)  # main() writes benchmarks/ under cwd
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"")
        ref = tmp_path / "clip.vtt"
        ref.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world\n")
        monkeypatch.setattr(
            benchmark.TranscriptionService,
            "transcribe_file",
            lambda self, *a, **kw: {"segments": [(0.0, 1.0, "hello world", None)]},
        )

        def run(*extra_args):
            assert benchmark.main([str(audio), "--reference", str(ref), *extra_args]) == 0
            out = capsys.readouterr()
            return json.loads(out.out), out.err

        return run

    @pytest.fixture
    def apple_silicon_default(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")

    def test_defaulted_parakeet_is_reported_as_parakeet(
        self, monkeypatch, apple_silicon_default, run_benchmark
    ):
        # Make the parakeet-mlx import succeed regardless of the test venv.
        monkeypatch.setitem(sys.modules, "parakeet_mlx", types.ModuleType("parakeet_mlx"))
        report, progress = run_benchmark()
        assert report["engine"] == "parakeet"
        assert "engine=parakeet" in progress

    def test_parakeet_fallback_is_reported_as_whisper(
        self, monkeypatch, apple_silicon_default, run_benchmark
    ):
        monkeypatch.setitem(sys.modules, "parakeet_mlx", None)  # import raises
        report, progress = run_benchmark()
        assert report["engine"] == "whisper"
        assert "engine=whisper" in progress

    def test_explicit_engine_flag_is_reported(self, apple_silicon_default, run_benchmark):
        report, _ = run_benchmark("--engine", "whisper")
        assert report["engine"] == "whisper"

    def test_parakeet_run_reports_the_parakeet_model(
        self, monkeypatch, apple_silicon_default, run_benchmark
    ):
        monkeypatch.setitem(sys.modules, "parakeet_mlx", types.ModuleType("parakeet_mlx"))
        monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
        monkeypatch.setenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
        report, _ = run_benchmark()
        assert report["model"] == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_whisper_run_reports_the_whisper_model(self, monkeypatch, run_benchmark):
        monkeypatch.setenv("WHISPER_MODEL", "small")
        report, _ = run_benchmark("--engine", "whisper")
        assert report["model"] == "small"

    def test_model_flag_reports_that_model(self, run_benchmark):
        report, _ = run_benchmark("--engine", "whisper", "--model", "tiny")
        assert report["model"] == "tiny"
