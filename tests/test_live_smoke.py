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

"""Tests for the live smoke surface of jointfm_client."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
import pytest

from jointfm_client import (
    DATAROBOT_API_TOKEN_ENV,
    DATAROBOT_ENDPOINT_ENV,
    JointFMClient,
    JOINTFM_DEPLOYMENT_ID_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
)

DATAROBOT_DEPLOYMENT_ID_ENV = "DATAROBOT_DEPLOYMENT_ID"
_REQUIRED_LIVE_ENV_NAMES = (
    DATAROBOT_ENDPOINT_ENV,
    DATAROBOT_API_TOKEN_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
)


def test_live_datarobot_health_smoke() -> None:
    """Live datarobot health smoke."""
    dotenv_path = Path(".env")
    dotenv_values_map = _dotenv_strings(dotenv_path)
    merged_env = dict(dotenv_values_map)
    merged_env.update(os.environ)

    deployment_id = merged_env.get(DATAROBOT_DEPLOYMENT_ID_ENV)
    if not isinstance(deployment_id, str) or deployment_id.strip() == "":
        pytest.skip(
            "Set DATAROBOT_DEPLOYMENT_ID in the environment or .env to run the live hosted smoke test"
        )

    missing = [
        name
        for name in _REQUIRED_LIVE_ENV_NAMES
        if not isinstance(merged_env.get(name), str) or merged_env[name].strip() == ""
    ]
    if missing:
        pytest.skip(
            "Live hosted smoke test requires "
            + ", ".join(missing)
            + " in the environment or .env"
        )

    client = JointFMClient.from_env(
        env={JOINTFM_DEPLOYMENT_ID_ENV: deployment_id.strip()},
        dotenv_path=dotenv_path if dotenv_path.exists() else None,
    )

    health = client.health()

    assert health.schema_version == client.settings.schema_version
    if client.settings.model_version is not None:
        assert health.model_version == client.settings.model_version


def _dotenv_strings(dotenv_path: Path) -> dict[str, str]:
    """Dotenv strings."""
    if not dotenv_path.exists():
        return {}

    return {
        key: value
        for key, value in dotenv_values(dotenv_path).items()
        if isinstance(key, str) and isinstance(value, str)
    }
