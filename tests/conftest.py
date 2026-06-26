"""Shared pytest fixtures for the jointfm_client test suite."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json_fixture(name: str) -> dict[str, Any]:
    """Load json fixture."""
    with (_FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


@pytest.fixture
def json_fixture_loader() -> Callable[[str], dict[str, Any]]:
    """Json fixture loader."""
    return _load_json_fixture
