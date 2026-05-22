from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json_fixture(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


@pytest.fixture
def json_fixture_loader() -> Callable[[str], dict[str, Any]]:
    return _load_json_fixture