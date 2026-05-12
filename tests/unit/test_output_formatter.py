import json

from src.config import Config
from src.output.formatter import OutputFormatter


def test_pretty_output_merges_adjacent_segments_by_speaker(tmp_path):
    config = Config(output_format="pretty")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "pretty.txt"

    segments = [
        (0.0, 1.5, "I know she would have been honored", "SPEAKER_00"),
        (1.6, 3.0, "to receive this award.", "SPEAKER_00"),
        (6.5, 8.0, "Thank you.", "SPEAKER_00"),
    ]

    formatter.save_transcript(segments, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "I know she would have been honored to receive this award." in content
    assert "Thank you." in content
    assert content.count("SPEAKER_00") == 2


def test_pretty_output_keeps_speaker_breaks(tmp_path):
    config = Config(output_format="pretty")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "pretty.txt"

    segments = [
        (0.0, 1.0, "Hello there.", "SPEAKER_00"),
        (1.1, 2.0, "Hi.", "SPEAKER_01"),
    ]

    formatter.save_transcript(segments, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "SPEAKER_00" in content
    assert "SPEAKER_01" in content
    assert "Hello there." in content
    assert "Hi." in content


# ---------------------------------------------------------------------------
# vtt-voice (WebVTT with `<v Speaker>` voice spans)
# ---------------------------------------------------------------------------


def test_vtt_voice_wraps_text_in_voice_tag(tmp_path):
    config = Config(output_format="vtt-voice")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.vtt"

    segments = [
        (0.0, 1.5, "Hello there.", "SPEAKER_00"),
        (1.6, 3.0, "General Kenobi.", "SPEAKER_01"),
    ]
    formatter.save_transcript(segments, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert content.startswith("WEBVTT")
    assert "<v SPEAKER_00>Hello there.</v>" in content
    assert "<v SPEAKER_01>General Kenobi.</v>" in content
    # No `Speaker: text` legacy form should remain.
    assert "SPEAKER_00: " not in content
    assert "SPEAKER_01: " not in content


def test_vtt_voice_omits_tag_when_speaker_empty(tmp_path):
    config = Config(output_format="vtt-voice")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.vtt"

    formatter.save_transcript([(0.0, 1.0, "plain narration", "")], str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "<v" not in content
    assert "plain narration" in content


def test_vtt_voice_escapes_html_in_cue_text(tmp_path):
    config = Config(output_format="vtt-voice")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.vtt"

    formatter.save_transcript(
        [(0.0, 1.0, "2 < 3 & 4 > 1", "SPEAKER_00")],
        str(output_path),
    )
    content = output_path.read_text(encoding="utf-8")

    assert "2 &lt; 3 &amp; 4 &gt; 1" in content
    # The literal characters must not appear inside cue payload.
    assert "<v SPEAKER_00>2 < 3" not in content


def test_vtt_voice_sanitizes_voice_id():
    config = Config(output_format="vtt-voice")
    formatter = OutputFormatter(config)

    assert formatter._vtt_voice_id("Alice") == "Alice"
    assert formatter._vtt_voice_id("Alice   <Smith>") == "Alice Smith"
    assert formatter._vtt_voice_id("   ") == "Speaker"
    assert formatter._vtt_voice_id("\nMulti\nline") == "Multi line"


def test_vtt_voice_timestamp_uses_period_separator(tmp_path):
    config = Config(output_format="vtt-voice")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.vtt"

    formatter.save_transcript(
        [(0.0, 1.5, "Hello.", "SPEAKER_00")],
        str(output_path),
    )
    content = output_path.read_text(encoding="utf-8")

    # WebVTT uses `HH:MM:SS.mmm` (period, not comma like SRT).
    assert "00:00:00.000 --> 00:00:01.500" in content


# ---------------------------------------------------------------------------
# json3 (YouTube auto-caption wire format)
# ---------------------------------------------------------------------------


def test_json3_emits_wire_magic_and_events(tmp_path):
    config = Config(output_format="json3")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.json3"

    segments = [
        (0.0, 1.5, "First cue", ""),
        (1.6, 3.0, "Second cue", ""),
    ]
    formatter.save_transcript(segments, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["wireMagic"] == "pb3"
    assert len(data["events"]) == 2
    for event in data["events"]:
        assert "tStartMs" in event
        assert "dDurationMs" in event
        assert isinstance(event["segs"], list)
        assert "utf8" in event["segs"][0]


def test_json3_timestamps_in_integer_ms(tmp_path):
    config = Config(output_format="json3")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.json3"

    segments = [(1.234, 2.567, "test", "")]
    formatter.save_transcript(segments, str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    event = data["events"][0]
    assert isinstance(event["tStartMs"], int)
    assert event["tStartMs"] == 1234
    assert isinstance(event["dDurationMs"], int)
    assert event["dDurationMs"] == 1333  # 2567 - 1234


def test_json3_duration_never_negative(tmp_path):
    config = Config(output_format="json3")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.json3"

    # Degenerate segment where end <= start (e.g. from a zero-length VAD chunk).
    formatter.save_transcript(
        [(5.0, 4.999, "shouldnt happen but", "")],
        str(output_path),
    )
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["events"][0]["dDurationMs"] == 0


def test_json3_prefixes_speaker_on_utf8(tmp_path):
    config = Config(output_format="json3")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "out.json3"

    formatter.save_transcript(
        [(0.0, 1.0, "hello", "SPEAKER_00"), (1.0, 2.0, "narrator", "")],
        str(output_path),
    )
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["events"][0]["segs"][0]["utf8"] == "[SPEAKER_00] hello"
    # No speaker → no prefix.
    assert data["events"][1]["segs"][0]["utf8"] == "narrator"


def test_unsupported_format_raises_valueerror(tmp_path):
    config = Config(output_format="json")
    formatter = OutputFormatter(config)
    formatter.format = "totally-made-up"

    try:
        formatter.save_transcript([(0.0, 1.0, "x", "")], str(tmp_path / "f"))
    except ValueError as exc:
        assert "totally-made-up" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown format")
