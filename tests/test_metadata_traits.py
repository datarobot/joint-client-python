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

"""Tests for expert-hint metadata traits in jointfm_client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    HealthMetadata,
    JointFMClient,
    UnsupportedServiceContractError,
)

_MODEL_VERSION = "jointfm-inference:0.3.0+ckpt.smoke-1"


def _health(traits: list[str] | None) -> dict[str, Any]:
    """Health payload that advertises ``traits``, or omits the field for ``None``."""
    payload: dict[str, Any] = {
        "status": "ok",
        "schema_version": "v5",
        "image_version": "0.3.0",
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "smoke-1",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "gmm",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": ["forecast"],
        "supported_condition_kinds": [],
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "max_sample_count": 64,
        "max_concurrent_requests": 1,
    }
    if traits is not None:
        payload["supported_metadata_traits"] = traits
    return payload


def _mean_response(request: Mapping[str, Any]) -> dict[str, Any]:
    """Mean response that answers ``request`` with zeros."""
    return {
        "schema_version": "v5",
        "image_version": "0.3.0",
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "smoke-1",
        "head": "gmm",
        "query_mode": "forecast",
        "return_mode": "mean",
        "outputs": {
            "query_times": list(request["query_times"]),
            "requested_columns": list(request["requested_columns"]),
            "mean": [[0.0] * len(request["requested_columns"])],
            "samples": None,
            "quantiles": None,
        },
        "diagnostics": {"history_rows": 2, "horizon_count": 1, "seed": None},
        "errors": [],
    }


class _Transport:
    """Serve one health payload and record every forecast request."""

    def __init__(self, health: dict[str, Any]) -> None:
        self.health = health
        self.forecasts: list[Mapping[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Answer the health probe of a client configured without settings."""
        del url
        return self.health

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record and answer one forecast request."""
        del url
        self.forecasts.append(payload)
        return _mean_response(payload)


def _forecast(transport: _Transport) -> None:
    """Send one mean forecast whose target carries a jump-process hint."""
    url = "https://example.com/predict"
    client = JointFMClient(health_url=url, predict_url=url, transport=transport)
    client.forecast_mean(
        [{"target": 1.0}, {"target": 2.0}],
        schema=DataFrameSchema(
            columns=(
                ColumnSpec(
                    name="target",
                    modality="numeric",
                    role="target",
                    traits={"jump_process": "present"},
                ),
            ),
            time_index_mode="ordinal",
        ),
        query_times=[2],
        requested_columns=["target"],
        model_version=_MODEL_VERSION,
    )


def test_traits_travel_in_the_column_descriptor() -> None:
    """A hinted column sends its traits; an unhinted one sends no field."""
    hinted = ColumnSpec(name="y", modality="numeric", traits={"seasonal": "present"})

    assert hinted.to_payload()["traits"] == {"seasonal": "present"}
    assert "traits" not in ColumnSpec(name="y", modality="numeric").to_payload()


@pytest.mark.parametrize("traits", [{"seasonal": True}, {3: "present"}, ["seasonal"]])
def test_trait_mappings_must_hold_strings(traits: Any) -> None:
    """The client checks the hint's shape; the deployment owns the vocabulary."""
    with pytest.raises(ValueError, match="columns.traits"):
        ColumnSpec(name="y", modality="numeric", traits=traits)


def test_health_without_the_field_supports_no_traits() -> None:
    """A deployment that predates hints drops them, so it advertises none."""
    assert HealthMetadata.from_payload(_health(None)).supported_metadata_traits == ()
    assert HealthMetadata.from_payload(
        _health(["jump_process", "seasonal"])
    ).supported_metadata_traits == ("jump_process", "seasonal")


@pytest.mark.parametrize("advertised", [None, [], ["seasonal"]])
def test_hints_the_deployment_does_not_read_are_refused_before_sending(
    advertised: list[str] | None,
) -> None:
    """No forecast request leaves the client with a hint that would vanish."""
    transport = _Transport(_health(advertised))

    with pytest.raises(UnsupportedServiceContractError, match="jump_process"):
        _forecast(transport)

    assert transport.forecasts == []


def test_hints_the_deployment_reads_are_sent() -> None:
    """An advertised trait reaches the request the deployment receives."""
    transport = _Transport(_health(["jump_process"]))

    _forecast(transport)

    [sent] = transport.forecasts
    assert sent["columns"][0]["traits"] == {"jump_process": "present"}
