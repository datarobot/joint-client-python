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
    JointFMRequestError,
    UnsupportedModelVersionError,
    UnsupportedServiceContractError,
)


def _instance(deployment_id: str) -> JointFMInstanceSettings:
    """Instance."""
    url = (
        "https://app.datarobot.com/api/v2/deployments/"
        f"{deployment_id}/predictionsUnstructured"
    )
    return JointFMInstanceSettings(
        deployment_id=deployment_id,
        predict_url=url,
        health_url=url,
    )


def _health_payload(
    *,
    model_version: str = "jointfm-inference:0.2.0+ckpt.sdk-test",
    checkpoint_version: str = "sdk-test",
    max_sample_count: int = 4096,
) -> dict[str, object]:
    """Health payload."""
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


class _PoolTransport:
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
        self.urls: list[str] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        raise AssertionError(f"unexpected GET {url}")

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        self.urls.append(url)
        deployment_id = url.rstrip("/").split("/")[-2]
        if deployment_id in self.fail_ids:
            if self.fail_status < 0:
                raise JointFMRequestError(f"{deployment_id} unreachable")
            raise JointFMHTTPStatusError(
                f"{deployment_id} unavailable",
                status_code=self.fail_status,
                response_body_excerpt="unavailable",
            )
        if payload.get("request_type") == "health":
            return self.health_by_id.get(deployment_id, _health_payload())
        return {"ok": True, "deployment_id": deployment_id}


def test_pool_round_robin_and_retries_next_instance_on_470() -> None:
    """Pool round robin and retries next instance on 470."""
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_PoolTransport(fail_ids=frozenset({"a"}), fail_status=470),
    )

    assert pool.next_instance().deployment_id == "a"
    assert pool.next_instance().deployment_id == "b"
    assert pool.next_instance().deployment_id == "a"

    result = pool.post_json({"schema_version": "v1"})

    assert result == {"ok": True, "deployment_id": "b"}


def test_pool_raises_when_all_instances_unavailable() -> None:
    """Pool raises when all instances unavailable."""
    transport = _PoolTransport(fail_ids=frozenset({"a", "b"}), fail_status=470)
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=transport,
    )

    with pytest.raises(JointFMHTTPStatusError, match="unavailable"):
        pool.post_json({"schema_version": "v1"})


def test_pool_health_rejects_model_or_checkpoint_mismatch() -> None:
    """Pool health rejects model or checkpoint mismatch."""
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_PoolTransport(
            health_by_id={
                "a": _health_payload(),
                "b": _health_payload(
                    model_version="jointfm-inference:9.9.9+ckpt.other"
                ),
            }
        ),
    )
    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        pool.probe_all_health()

    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_PoolTransport(
            health_by_id={
                "a": _health_payload(checkpoint_version="ckpt-a"),
                "b": _health_payload(checkpoint_version="ckpt-b"),
            }
        ),
    )
    with pytest.raises(UnsupportedServiceContractError, match="checkpoint_version"):
        pool.probe_all_health()


def test_pool_health_uses_minimum_max_sample_count() -> None:
    """Pool health uses minimum max sample count."""
    pool = JointFMInstancePool(
        instances=(_instance("a"), _instance("b")),
        transport=_PoolTransport(
            health_by_id={
                "a": _health_payload(max_sample_count=100),
                "b": _health_payload(max_sample_count=40),
            }
        ),
    )

    metadata = pool.probe_all_health()

    assert metadata.max_sample_count == 40
