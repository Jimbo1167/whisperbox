"""`transcribe.py completion` prints a shell completion script.

Regression: it shelled out to sys.argv[0] with _TRANSCRIBE_COMPLETE set, but
Click derives the variable from the program name (_TRANSCRIBE_PY_COMPLETE), so
it printed nothing (or failed outright under the system Python).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.transcribe import cli  # noqa: E402


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_prints_a_completion_script_for_each_shell(shell):
    result = CliRunner().invoke(cli, ["completion", shell])

    assert result.exit_code == 0, result.output
    assert "_TRANSCRIBE_PY_COMPLETE" in result.output
    assert "transcribe.py" in result.output


def test_shell_defaults_to_the_login_shell(monkeypatch):
    monkeypatch.setenv("SHELL", "/opt/homebrew/bin/fish")

    result = CliRunner().invoke(cli, ["completion"])

    assert result.exit_code == 0, result.output
    assert "fish_complete" in result.output


def test_unsupported_shell_is_a_usage_error(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/tcsh")

    result = CliRunner().invoke(cli, ["completion"])

    assert result.exit_code == 2
