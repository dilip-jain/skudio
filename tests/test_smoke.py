"""Smoke tests for the package skeleton"""

from __future__ import annotations

import re

import skudio


def test_version_is_semver_like() -> None:
    """Test: check version format."""
    assert re.match(r"^\d+\.\d+\.\d+", skudio.__version__)


def test_name_and_tagline() -> None:
    """Test: check name & tagline."""
    assert skudio.NAME
    assert skudio.TAGLINE
