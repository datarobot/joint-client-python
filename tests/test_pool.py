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

"""Tests for JointFM instance pool load balancing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jointfm_client import (
    JointFMHTTPStatusError,
    JointFMInstancePool,
    JointFMInstanceSettings,
    UnsupportedModelVersionError,
    UnsupportedServiceContractError,
)


def _instance(deployment_id: str) -> JointFMInstanceSettings:
    """Instance."""
    return JointFMInstanceSettings(
        deployment_id=deployment_id,
        predict_url=(
            "https://app.datarobot.com/api/v2/deployments/"
            f"{deployment_id}/predictionsUnstructured"
        ),
    )


def _health(
    *,
    model_version: str = "jointfm-inference:0.2.0+ckpt.sdk-test",
    checkpoint_version: str = "sdk-test",
    max_sample_count: int = 4096,
) -> dict[str, object]:
    """Health."""
    return {
        "status": "ok",
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": model_version,
        "checkpoint_version": checkpoint_version,
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "studentt",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": ["forecast"],
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "max_sample_count": max_sample_count,
    }


class _Transport:
    """Transport that serves health/predict per deployment id."""

    def __init__(
        self,
        *,
        health_by_id: Mapping[str, Mapping[str, Any]] | None = None,
        fail_ids: frozenset[str] = frozenset(),
        fail_status: int = 470,
    ) -> None:
        self.health_by_id = dict(health_by_id or {})
        self.fail_ids = fail_ids
        self.fail_status = fail_status

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        raise AssertionError(f"unexpected GET {url}")

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        deployment_id = url.rstrip("/").split("/")[-2]
        if deployment_id in self.fail_ids:
            raise JointFMHTTPStatusError(
                f"{deployment_id} unavailable",
                status_code=self.fail_status,
                response_body_excerpt="unavailable",
            )
        if payload.get("request_type") == "health":
            return self.health_by_id.get(deployment_id, _health())
        return {"ok": True, "deployment_id": deployment_id}


def test_pool_retries_next_instance_on_470() -> None:
    """Pool retries next instance on 470."""
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_Transport(fail_ids=frozenset({"a"})),
    )
    assert pool.next_instance().deployment_id == "a"
    assert pool.next_instance().deployment_id == "b"
    assert pool.post_json({"schema_version": "v1"}) == {
        "ok": True,
        "deployment_id": "b",
    }


def test_pool_raises_when_all_instances_unavailable() -> None:
    """Pool raises when all instances unavailable."""
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_Transport(fail_ids=frozenset({"a", "b"})),
    )
    with pytest.raises(JointFMHTTPStatusError, match="unavailable"):
        pool.post_json({"schema_version": "v1"})


def test_pool_health_rejects_mismatch_and_aligns_sample_cap() -> None:
    """Pool health rejects mismatch and aligns sample cap."""
    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        JointFMInstancePool(
            instances=(_instance("a"), _instance("b")),
            transport=_Transport(
                health_by_id={
                    "a": _health(),
                    "b": _health(model_version="jointfm-inference:9.9.9+ckpt.other"),
                }
            ),
        ).probe_all_health()

    with pytest.raises(UnsupportedServiceContractError, match="checkpoint_version"):
        JointFMInstancePool(
            instances=(_instance("a"), _instance("b")),
            transport=_Transport(
                health_by_id={
                    "a": _health(checkpoint_version="ckpt-a"),
                    "b": _health(checkpoint_version="ckpt-b"),
                }
            ),
        ).probe_all_health()

    metadata = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_Transport(
            health_by_id={
                "a": _health(max_sample_count=100),
                "b": _health(max_sample_count=40),
            }
        ),
    ).probe_all_health()
    assert metadata.max_sample_count == 40
