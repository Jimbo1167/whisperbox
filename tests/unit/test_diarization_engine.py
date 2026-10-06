import sys
import types
from types import SimpleNamespace

import pytest

import src.diarization.engine as engine_module
from src.config import Config
from src.diarization.engine import DiarizationEngine, _check_torchcodec_available


class _FakeAnnotation:
    def __init__(self):
        self._tracks = [
            (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]

    def itertracks(self, yield_label=False):
        return self._tracks


def test_unwrap_diarization_annotation_direct():
    config = Config(include_diarization=False)
    engine = DiarizationEngine(config, test_mode=True)
    annotation = _FakeAnnotation()

    assert engine._unwrap_diarization_result(annotation) is annotation


def test_unwrap_diarization_output_object():
    config = Config(include_diarization=False)
    engine = DiarizationEngine(config, test_mode=True)
    annotation = _FakeAnnotation()
    diarize_output = SimpleNamespace(speaker_diarization=annotation)

    assert engine._unwrap_diarization_result(diarize_output) is annotation


def _break_torchcodec(monkeypatch):
    """Make ``import torchcodec`` fail regardless of the host environment."""
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)


def test_torchcodec_probe_raises_actionable_error(monkeypatch):
    """Regression: docs/bugs/2026-08-30-diarization-audiodecoder.md (Issue A).

    A broken torchcodec used to surface only as pyannote's swallowed import
    warning, then a NameError mid-job. The probe must fail with a message
    that names the FFmpeg/torchcodec mismatch and a fix.
    """
    _break_torchcodec(monkeypatch)

    with pytest.raises(RuntimeError, match="FFmpeg"):
        _check_torchcodec_available()


def test_torchcodec_probe_detects_lazy_load_failure(monkeypatch):
    """torchcodec >= 0.12 imports fine without FFmpeg and only fails on first
    decoder use; the probe must catch that mode too."""
    fake_decoders = types.ModuleType("torchcodec.decoders")

    class BrokenAudioDecoder:
        def __init__(self, source):
            raise OSError("Could not load this library: libtorchcodec_core9.dylib")

    fake_decoders.AudioDecoder = BrokenAudioDecoder
    fake_pkg = types.ModuleType("torchcodec")
    fake_pkg.decoders = fake_decoders
    monkeypatch.setitem(sys.modules, "torchcodec", fake_pkg)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_decoders)

    with pytest.raises(RuntimeError, match="FFmpeg"):
        _check_torchcodec_available()


@pytest.mark.env
def test_torchcodec_decodes_on_this_machine():
    """Environment assertion: the installed torchcodec must decode against the
    system FFmpeg. Fails after e.g. a `brew upgrade ffmpeg` past the majors the
    installed torchcodec supports — the silent re-break called out in
    docs/bugs/2026-08-30-diarization-audiodecoder.md.
    """
    pytest.importorskip("torchcodec")
    _check_torchcodec_available()


def test_load_model_fails_fast_before_pipeline_load(monkeypatch):
    """The probe fires before the (slow) pyannote pipeline download/load."""
    _break_torchcodec(monkeypatch)

    class ExplodingPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            pytest.fail("Pipeline load attempted despite broken torchcodec")

    monkeypatch.setattr(engine_module, "Pipeline", ExplodingPipeline)

    config = Config(include_diarization=True)
    config.hf_token = "test_token"
    engine = DiarizationEngine(config, test_mode=False)

    with pytest.raises(RuntimeError, match="torchcodec"):
        engine.ensure_model_loaded()


def test_load_model_skips_probe_in_test_mode(monkeypatch):
    """Mock diarizer path must stay usable on machines without torchcodec."""
    _break_torchcodec(monkeypatch)

    config = Config(include_diarization=True)
    engine = DiarizationEngine(config, test_mode=True)

    assert engine.diarizer is not None
