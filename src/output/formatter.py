import os
import json
import logging
import re
from typing import List, Tuple, Dict, Any, Optional
import math

from ..config import Config

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Handles formatting and saving transcripts in different formats."""
    
    def __init__(self, config: Config):
        """Initialize the output formatter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.format = config.output_format
    
    def save_transcript(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcription to file in the specified format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
            
        Raises:
            ValueError: If the output format is not supported
            IOError: If there's an error writing the file
        """
        logger.info(f"Saving transcript in {self.format} format to {output_path}")
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            if self.format == "txt":
                self._save_txt(segments, output_path)
            elif self.format == "srt":
                self._save_srt(segments, output_path)
            elif self.format == "vtt":
                self._save_vtt(segments, output_path)
            elif self.format == "vtt-voice":
                self._save_vtt_voice(segments, output_path)
            elif self.format == "json":
                self._save_json(segments, output_path)
            elif self.format == "json3":
                self._save_json3(segments, output_path)
            elif self.format == "pretty":
                self._save_pretty(segments, output_path)
            else:
                raise ValueError(f"Unsupported output format: {self.format}")
                
            logger.info(f"Transcript saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise

    def format_transcript(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """Format transcript content as a string for previews or console display."""
        if self.format == "txt":
            return self._format_txt(segments)
        if self.format == "srt":
            return self._format_srt(segments)
        if self.format == "vtt":
            return self._format_vtt(segments)
        if self.format == "vtt-voice":
            return self._format_vtt_voice(segments)
        if self.format == "json":
            return self._format_json(segments)
        if self.format == "json3":
            return self._format_json3(segments)
        if self.format == "pretty":
            return self._format_pretty(segments)
        raise ValueError(f"Unsupported output format: {self.format}")
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format seconds as timestamp.
        
        Args:
            seconds: Time in seconds
            vtt: Whether to use VTT format (with milliseconds)
            
        Returns:
            Formatted timestamp string
        """
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        seconds = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
    
    def _save_txt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in plain text format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_txt(segments))
    
    def _save_srt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in SRT format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_srt(segments))
    
    def _save_vtt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in WebVTT format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_vtt(segments))
    
    def _save_json(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in JSON format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_json(segments))

    def _format_txt(self, segments: List[Tuple[float, float, str, str]]) -> str:
        lines = []
        for start, end, text, speaker in segments:
            timestamp = f"[{self._format_timestamp(start, False)} --> {self._format_timestamp(end, False)}]"
            if speaker:
                lines.append(f"{timestamp} {speaker}: {text}")
            else:
                lines.append(f"{timestamp} {text}")
        return "\n".join(lines)

    def _format_srt(self, segments: List[Tuple[float, float, str, str]]) -> str:
        lines = []
        for i, (start, end, text, speaker) in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}")
            lines.append(f"{speaker}: {text}" if speaker else text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _format_vtt(self, segments: List[Tuple[float, float, str, str]]) -> str:
        lines = ["WEBVTT", ""]
        for i, (start, end, text, speaker) in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{self._format_timestamp(start, True)} --> {self._format_timestamp(end, True)}")
            lines.append(f"{speaker}: {text}" if speaker else text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _escape_vtt_text(self, text: str) -> str:
        # WebVTT cue payload only requires ``&``, ``<``, and ``>`` to be
        # encoded so they're not parsed as cue tags.
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _vtt_voice_id(self, speaker: str) -> str:
        # Voice IDs cannot contain ``<``, ``>``, or newlines.
        cleaned = re.sub(r"\s+", " ", speaker.replace("<", "").replace(">", "")).strip()
        return cleaned or "Speaker"

    def _format_vtt_voice(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """WebVTT with ``<v Speaker>...</v>`` voice spans (YouTube-style)."""
        lines = ["WEBVTT", ""]
        for i, (start, end, text, speaker) in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(
                f"{self._format_timestamp(start, True)} --> {self._format_timestamp(end, True)}"
            )
            escaped = self._escape_vtt_text(text)
            if speaker:
                voice = self._vtt_voice_id(speaker)
                lines.append(f"<v {voice}>{escaped}</v>")
            else:
                lines.append(escaped)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _save_vtt_voice(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_vtt_voice(segments))

    def _format_json(self, segments: List[Tuple[float, float, str, str]]) -> str:
        json_data = []
        for start, end, text, speaker in segments:
            json_data.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker
            })
        return json.dumps(json_data, ensure_ascii=False, indent=2)

    def _format_json3(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """YouTube auto-caption wire format ("json3"), consumable by yt-dlp.

        Schema::

            {"wireMagic": "pb3",
             "events": [
                 {"tStartMs": <int>, "dDurationMs": <int>,
                  "segs": [{"utf8": "..."}]},
                 ...
             ]}

        Speaker labels (when present) are prefixed onto the utf8 payload —
        the json3 wire format itself has no speaker field. Round-trip back to
        vtt/srt via ``yt-dlp --convert-subs vtt``.
        """
        events = []
        for start, end, text, speaker in segments:
            start_ms = int(round(start * 1000))
            end_ms = int(round(end * 1000))
            duration_ms = max(0, end_ms - start_ms)
            payload = f"[{speaker}] {text}" if speaker else text
            events.append({
                "tStartMs": start_ms,
                "dDurationMs": duration_ms,
                "segs": [{"utf8": payload}],
            })
        return json.dumps(
            {"wireMagic": "pb3", "events": events},
            ensure_ascii=False,
            indent=2,
        )

    def _save_json3(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_json3(segments))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _starts_as_continuation(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return True

        first_word = normalized.split()[0].lower().strip("\"'([{")
        continuation_words = {
            "and", "but", "or", "so", "because", "that", "which", "who",
            "then", "than", "if", "when", "while", "where", "as", "though",
            "although", "though", "however", "yet", "also", "still",
        }

        return normalized[:1].islower() or first_word in continuation_words

    def _ends_as_continuation(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return True
        return normalized.endswith((",", ";", ":", "-", "—"))

    def _join_text(self, left: str, right: str) -> str:
        left = self._normalize_text(left)
        right = self._normalize_text(right)
        if not left:
            return right
        if not right:
            return left

        if self._ends_as_continuation(left):
            return f"{left} {right}"
        if self._starts_as_continuation(right):
            return f"{left} {right}"
        return f"{left} {right}"

    def _group_pretty_segments(
        self, segments: List[Tuple[float, float, str, str]]
    ) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []

        for start, end, text, speaker in segments:
            text = self._normalize_text(text)
            if not text:
                continue

            if not groups:
                groups.append(
                    {"start": start, "end": end, "speaker": speaker, "text": text}
                )
                continue

            previous = groups[-1]
            gap = start - previous["end"]
            same_speaker = previous["speaker"] == speaker
            should_merge = (
                same_speaker and (
                    gap <= 1.2
                    or self._ends_as_continuation(previous["text"])
                    or self._starts_as_continuation(text)
                )
            )

            if should_merge:
                previous["end"] = end
                previous["text"] = self._join_text(previous["text"], text)
            else:
                groups.append(
                    {"start": start, "end": end, "speaker": speaker, "text": text}
                )

        return groups

    def _format_pretty(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """Format transcript in a speaker-aware readable paragraph format."""
        groups = self._group_pretty_segments(segments)
        blocks = []
        for group in groups:
            timestamp = (
                f"[{self._format_timestamp(group['start'], False)}"
                f" --> {self._format_timestamp(group['end'], False)}]"
            )
            heading = f"{timestamp} {group['speaker']}".rstrip()
            blocks.append(f"{heading}\n{group['text']}")
        return "\n\n".join(blocks)

    def _save_pretty(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in a speaker-aware readable paragraph format."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._format_pretty(segments))
    
    def format_transcript_for_display(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """Format transcript for display in the console.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            
        Returns:
            Formatted transcript string
        """
        return self.format_transcript(segments)
