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

"""Tests for the ``condition`` query mode on the client side.

Three surfaces are covered, and the split matters. The condition objects
validate what is wrong independently of any deployment, so a caller learns it
without a round trip. The capability gate refuses what *this* deployment has
not advertised, which is the only way to know before paying for a request.
The response parser reads back the two numbers the service reports and never
refuses on, so acting on them stays the caller's decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jointfm_client import (
    ColumnSpec,
    ConditionBlock,
    ConditionPlausibility,
    DataFrameSchema,
    EqualityCondition,
    ForecastRequest,
    ForecastRequestMetadata,
    ForecastResponse,
    HealthMetadata,
    IntervalCondition,
    build_forecast_payload,
    require_condition_support,
)
from jointfm_client.contract import QueryMode
from jointfm_client.exceptions import UnsupportedServiceContractError

_MODEL_VERSION = "jointfm-inference:0.3.0+ckpt.sdk-test"


def _schema() -> DataFrameSchema:
    """Build one three-column ordinal schema for request-level tests."""
    return DataFrameSchema(
        columns=(
            ColumnSpec(name="driver", modality="numeric"),
            ColumnSpec(name="hedge", modality="numeric"),
            ColumnSpec(name="target", modality="numeric", role="target"),
        ),
        time_index_mode="ordinal",
    )


def _request(
    block: ConditionBlock | None,
    *,
    requested_columns: tuple[str, ...] | None = ("target",),
    query_mode: QueryMode = "condition",
) -> ForecastRequest:
    """Build one condition request against the two-column schema."""
    return ForecastRequest(
        metadata=ForecastRequestMetadata(
            model_version=_MODEL_VERSION,
            query_mode=query_mode,
        ),
        schema=_schema(),
        history_rows=(
            {"driver": 1.0, "hedge": 0.5, "target": 10.0},
            {"driver": 1.1, "hedge": 0.6, "target": 11.0},
        ),
        query_times=(2, 3),
        requested_columns=requested_columns,
        condition=block,
    )


def _health(
    *,
    query_modes: tuple[str, ...] = ("forecast", "condition"),
    condition_kinds: tuple[str, ...] = ("equality", "interval"),
) -> HealthMetadata:
    """Build one advertisement without going through a transport."""
    return HealthMetadata(
        status="ok",
        schema_version="v3",
        image_version="0.3.0",
        model_version=_MODEL_VERSION,
        checkpoint_version="sdk-test",
        checkpoint_path="/models/jointfm.pt",
        device="cpu",
        head="gmm",
        decoding_strategy="parallel_dense",
        supported_query_modes=query_modes,
        supported_condition_kinds=condition_kinds,
        supported_return_modes=("mean", "samples", "quantiles", "log_prob"),
        supported_time_index_modes=("ordinal",),
        time_index_encoding="ordinal",
        max_sample_count=10000,
    )


def test_an_equality_condition_rejects_a_value_it_cannot_pin() -> None:
    """A pin must be a finite number; anything else is not an event."""
    with pytest.raises(ValueError, match="finite"):
        EqualityCondition(column="driver", value=float("inf"))


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (None, None, "conditions nothing"),
        (2.0, 1.0, "lower < upper"),
        (float("nan"), 1.0, "finite"),
    ],
)
def test_an_interval_condition_rejects_a_range_that_bounds_nothing(
    lower: float | None, upper: float | None, message: str
) -> None:
    """An interval open on both sides, or backwards, describes no region."""
    with pytest.raises(ValueError, match=message):
        IntervalCondition(column="driver", lower=lower, upper=upper)


def test_a_block_rejects_two_conditions_on_one_column() -> None:
    """A column carries at most one condition, of either kind."""
    with pytest.raises(ValueError, match="more than one condition"):
        ConditionBlock(
            query_time_index=0,
            conditions=(
                EqualityCondition(column="driver", value=1.0),
                IntervalCondition(column="driver", lower=0.0, upper=1.0),
            ),
        )


def test_a_block_reports_the_kinds_a_deployment_must_advertise() -> None:
    """The gate needs the kinds, and each kind appears once however many columns use it."""
    block = ConditionBlock(
        query_time_index=0,
        conditions=(
            EqualityCondition(column="driver", value=1.0),
            IntervalCondition(column="target", lower=0.0, upper=None),
        ),
    )

    assert block.kinds == ("equality", "interval")
    assert block.pinned_columns == ("driver",)
    assert block.conditioned_columns == ("driver", "target")


def test_the_mode_and_the_block_must_agree() -> None:
    """One without the other means a different request than the caller wrote."""
    block = ConditionBlock(
        query_time_index=0,
        conditions=(EqualityCondition(column="driver", value=1.0),),
    )

    with pytest.raises(ValueError, match="go together"):
        _request(block, query_mode="forecast")
    with pytest.raises(ValueError, match="go together"):
        _request(None)


@pytest.mark.parametrize(
    ("block", "requested_columns", "message"),
    [
        (
            ConditionBlock(
                query_time_index=0,
                conditions=(EqualityCondition(column="absent", value=1.0),),
            ),
            ("target",),
            "undeclared columns",
        ),
        (
            ConditionBlock(
                query_time_index=9,
                conditions=(EqualityCondition(column="driver", value=1.0),),
            ),
            ("target",),
            "outside the 2 requested future positions",
        ),
        (
            ConditionBlock(
                query_time_index=0,
                conditions=(EqualityCondition(column="driver", value=1.0),),
            ),
            ("driver", "target"),
            "pinned columns",
        ),
        (
            ConditionBlock(
                query_time_index=0,
                conditions=(
                    EqualityCondition(column="driver", value=1.0),
                    EqualityCondition(column="hedge", value=0.5),
                    EqualityCondition(column="target", value=2.0),
                ),
            ),
            None,
            "log_prob",
        ),
        (
            ConditionBlock(
                query_time_index=0,
                conditions=(
                    EqualityCondition(column="driver", value=1.0),
                    EqualityCondition(column="hedge", value=0.5),
                    IntervalCondition(column="target", lower=0.0, upper=None),
                ),
            ),
            None,
            "at least one column unconditioned",
        ),
    ],
)
def test_a_request_is_checked_against_its_own_schema_before_any_round_trip(
    block: ConditionBlock, requested_columns: tuple[str, ...] | None, message: str
) -> None:
    """The caller learns which column name it got wrong without paying for a request."""
    with pytest.raises(ValueError, match=message):
        _request(block, requested_columns=requested_columns)


def test_an_interval_column_may_still_be_read_out() -> None:
    """An interval fixes a region, so the column's distribution inside it is an answer."""
    block = ConditionBlock(
        query_time_index=0,
        conditions=(IntervalCondition(column="driver", lower=0.5, upper=1.5),),
    )

    request = _request(block, requested_columns=("driver", "target"))

    assert request.to_payload()["requested_columns"] == ["driver", "target"]


def test_the_payload_carries_the_block_the_service_parses() -> None:
    """The wire form names its position once and each condition names its kind."""
    payload = build_forecast_payload(
        model_version=_MODEL_VERSION,
        schema=_schema(),
        history_rows=({"driver": 1.0, "hedge": 0.5, "target": 10.0},),
        query_times=(2, 3),
        requested_columns=("target",),
        query_mode="condition",
        condition=ConditionBlock(
            query_time_index=1,
            conditions=(
                EqualityCondition(column="driver", value=1.5),
                IntervalCondition(column="target", lower=None, upper=12.0),
            ),
        ),
    )

    assert payload["query_mode"] == "condition"
    assert payload["condition"] == {
        "query_time_index": 1,
        "conditions": [
            {"column": "driver", "kind": "equality", "value": 1.5},
            {"column": "target", "kind": "interval", "lower": None, "upper": 12.0},
        ],
    }


def test_the_gate_refuses_a_deployment_that_advertises_no_condition_mode() -> None:
    """Discovery before the request is the point: the error must not cost a round trip."""
    block = ConditionBlock(
        query_time_index=0,
        conditions=(EqualityCondition(column="driver", value=1.0),),
    )

    with pytest.raises(
        UnsupportedServiceContractError, match="does not serve condition"
    ):
        require_condition_support(
            _health(query_modes=("forecast",), condition_kinds=()), block
        )


def test_the_gate_refuses_a_kind_the_deployment_does_not_answer() -> None:
    """A deployment may serve one kind before the other; the block says which it needs."""
    block = ConditionBlock(
        query_time_index=0,
        conditions=(IntervalCondition(column="driver", lower=0.0, upper=1.0),),
    )

    with pytest.raises(UnsupportedServiceContractError, match=r"kinds \['interval'\]"):
        require_condition_support(_health(condition_kinds=("equality",)), block)

    require_condition_support(_health(), block)


def test_the_equality_response_reads_back_its_plausibility(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The density of the pinned values is what separates confident from absurd."""
    request_payload = json_fixture_loader("condition_mean_request")
    response = ForecastResponse.from_payload(
        json_fixture_loader("condition_mean_response"),
        request_payload=request_payload,
    )

    assert response.query_mode == "condition"
    assert response.plausibility == ConditionPlausibility(
        equality_log_density=-1.27,
        region_log_probability=None,
    )
    assert response.diagnostics.condition_draws == 500
    assert response.diagnostics.interval_estimator is None


def test_the_response_describes_the_conditioned_position_alone(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The request asked for two future rows; a condition answers about one."""
    request_payload = json_fixture_loader("condition_mean_request")
    assert request_payload["query_times"] == [2, 3]

    response = ForecastResponse.from_payload(
        json_fixture_loader("condition_mean_response"),
        request_payload=request_payload,
    )

    assert response.query_times == (3,)
    assert response.diagnostics.horizon_count == 1


def test_an_interval_response_reads_back_its_estimator_accuracy(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A multi-column region is estimated, and the caller judges the estimate itself."""
    response = ForecastResponse.from_payload(
        json_fixture_loader("condition_interval_response"),
        request_payload=json_fixture_loader("condition_mean_request"),
    )

    assert response.plausibility is not None
    assert response.plausibility.equality_log_density is None
    assert response.plausibility.region_log_probability == pytest.approx(-8.91)
    estimator = response.diagnostics.interval_estimator
    assert estimator is not None
    assert estimator.points == 16384
    assert estimator.effective_sample_size == pytest.approx(11453.2)
