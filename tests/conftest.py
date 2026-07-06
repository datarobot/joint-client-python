# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
