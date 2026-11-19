"""Test: Server"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from skudio.cli import _resolve_project_dir, main
from skudio.server.net import ensure_loopback, pick_free_port
from skudio.server.token import new_token, verify_token


def test_pick_free_port_returns_usable_port() -> None:
    """Test: Pick free port."""
    port = pick_free_port()
    assert 1024 <= port <= 65535


def test_ensure_loopback_accepts_localhost() -> None:
    """Test: Ensure loopback accepts localhost."""
    assert ensure_loopback("127.0.0.1") == "127.0.0.1"
    assert ensure_loopback("localhost") == "localhost"
    assert ensure_loopback("::1") == "::1"


def test_ensure_loopback_rejects_wildcard() -> None:
    """Test: Ensure loopback rejects wildcard."""
    with pytest.raises(ValueError):
        ensure_loopback("0.0.0.0")


def test_token_verifies_itself() -> None:
    """Test: Token verification."""
    t = new_token()
    assert verify_token(t, t)


def test_token_rejects_mismatch() -> None:
    """Test: Token rejection."""
    t = new_token()
    assert not verify_token(t, t + "a")
    assert not verify_token(t, "")


def test_token_length_reasonable() -> None:
    """Test: Token length."""
    t = new_token()
    assert len(t) >= 40


def test_resolve_project_dir_creates_directory(tmp_path: Path) -> None:
    """Test: Resolve project directory - create."""
    target = tmp_path / "proj"
    assert not target.exists()
    resolved = _resolve_project_dir(target)
    assert resolved == target.resolve()
    assert resolved.is_dir()


def test_resolve_project_dir_defaults_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test: Resolve project directory - defaults to home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows equivalent
    resolved = _resolve_project_dir(None)
    assert resolved.is_dir()
    assert resolved.name == "default"
    assert resolved.parent.name == "projects"


def test_cli_help_lists_project_dir_option() -> None:
    """Test: CLI help."""
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--project-dir" in result.output
    assert "Directory to persist runs" in result.output
