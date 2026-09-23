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

from collections.abc import Callable, Mapping
from typing import Any, cast

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
    JointFMClient,
    MeanForecastResult,
    SampleForecastResult,
    build_forecast_payload,
    build_forecast_payload_from_dataframe,
    require_condition_support,
    validate_service_metadata,
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
        schema_version="v4",
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


def test_a_pinned_column_may_be_read_out_in_any_position() -> None:
    """A projection is condition-agnostic, so a pin is nameable like any column.

    Its answer is the request's own value, which is what lets a scenario frame
    be compared against an unconditioned one column for column.
    """
    block = ConditionBlock(
        query_time_index=0,
        conditions=(EqualityCondition(column="driver", value=1.0),),
    )

    request = _request(block, requested_columns=("target", "driver"))

    assert request.to_payload()["requested_columns"] == ["target", "driver"]


def test_omitting_the_projection_states_nothing_on_the_wire() -> None:
    """The default lives in the service, so the client must not invent one.

    A client-side default would be a second copy of the rule, free to drift from
    the one the deployment actually applies.
    """
    block = ConditionBlock(
        query_time_index=0,
        conditions=(EqualityCondition(column="driver", value=1.0),),
    )

    payload = _request(block, requested_columns=None).to_payload()

    assert "requested_columns" not in payload


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


class _ConditionTransport:
    """Fake JSON transport for a deployment that advertises the condition mode.

    It records every predict payload and answers each sample request with as
    many draws as it asked for, so the batching path can be checked end to end
    without a service.
    """

    def __init__(
        self,
        *,
        health_payload: dict[str, Any],
        predict_payload: dict[str, Any] | None = None,
    ) -> None:
        """Remember the advertisement and the canned prediction to serve."""
        self.health_payload = health_payload
        self.predict_payload = predict_payload
        self.payloads: list[dict[str, Any]] = []
        self.health_count = 0

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Serve the health advertisement on the local health route."""
        assert url == "http://127.0.0.1:8080/healthz"
        self.health_count += 1
        return self.health_payload

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record the predict payload and answer it."""
        assert url == "http://127.0.0.1:8080/predict"
        self.payloads.append(dict(payload))
        if self.predict_payload is not None:
            return self.predict_payload
        sample_count = payload["n_samples"]
        assert isinstance(sample_count, int)
        start = sum(cast(int, earlier["n_samples"]) for earlier in self.payloads[:-1])
        return {
            "schema_version": "v4",
            "image_version": "0.3.0",
            "model_version": _MODEL_VERSION,
            "checkpoint_version": "sdk-test",
            "head": "gmm",
            "query_mode": "condition",
            "return_mode": "samples",
            "outputs": {
                "query_times": [3],
                "requested_columns": ["target"],
                "mean": None,
                "samples": [[[float(start + index)]] for index in range(sample_count)],
                "quantiles": None,
            },
            "plausibility": {
                "equality_log_density": -1.27,
                "region_log_probability": None,
            },
            "diagnostics": {
                "history_rows": 2,
                "horizon_count": 1,
                "seed": payload.get("seed"),
                "condition_draws": sample_count,
                "interval_estimator": None,
            },
            "errors": [],
        }


def _health_payload(
    *,
    query_modes: list[str],
    condition_kinds: list[str],
    max_sample_count: int = 4096,
) -> dict[str, Any]:
    """Build one health advertisement as the service serializes it."""
    return {
        "status": "ok",
        "schema_version": "v4",
        "image_version": "0.3.0",
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "sdk-test",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "gmm",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": query_modes,
        "supported_condition_kinds": condition_kinds,
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "max_sample_count": max_sample_count,
    }


def _client(transport: _ConditionTransport) -> JointFMClient:
    """Build a local-service client over the fake transport."""
    return JointFMClient(
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        transport=transport,
    )


_HISTORY_ROWS = (
    {"driver": 1.0, "hedge": 0.5, "target": 10.0},
    {"driver": 1.1, "hedge": 0.6, "target": 11.0},
)
_PIN_DRIVER = ConditionBlock(
    query_time_index=1,
    conditions=(EqualityCondition(column="driver", value=1.5),),
)


def test_the_client_sends_the_condition_and_reads_the_answer_back(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The typed helper carries the block onto the wire and types what comes back."""
    transport = _ConditionTransport(
        health_payload=_health_payload(
            query_modes=["forecast", "condition"],
            condition_kinds=["equality", "interval"],
        ),
        predict_payload=json_fixture_loader("condition_mean_response"),
    )

    result = _client(transport).forecast_mean(
        list(_HISTORY_ROWS),
        schema=_schema(),
        query_times=[2, 3],
        requested_columns=["target"],
        model_version=_MODEL_VERSION,
        seed=7,
        condition=_PIN_DRIVER,
    )

    assert isinstance(result, MeanForecastResult)
    assert result.query_mode == "condition"
    assert result.query_times == (3,)
    assert result.mean == ((13.5,),)
    assert result.plausibility == ConditionPlausibility(equality_log_density=-1.27)
    assert len(transport.payloads) == 1
    sent = transport.payloads[0]
    assert sent["query_mode"] == "condition"
    assert sent["condition"] == _PIN_DRIVER.to_payload()


def test_the_client_refuses_before_posting_when_the_deployment_cannot_condition() -> (
    None
):
    """The gate runs on the advertisement, so the predict route is never touched."""
    transport = _ConditionTransport(
        health_payload=_health_payload(query_modes=["forecast"], condition_kinds=[]),
    )

    with pytest.raises(UnsupportedServiceContractError, match="does not serve"):
        _client(transport).forecast_mean(
            list(_HISTORY_ROWS),
            schema=_schema(),
            query_times=[2, 3],
            requested_columns=["target"],
            model_version=_MODEL_VERSION,
            condition=_PIN_DRIVER,
        )

    assert transport.health_count == 1
    assert transport.payloads == []


def test_batched_condition_samples_merge_into_one_conditional_answer() -> None:
    """Every batch repeats the same block; the merge recounts the draws and keeps the plausibility."""
    transport = _ConditionTransport(
        health_payload=_health_payload(
            query_modes=["forecast", "condition"],
            condition_kinds=["equality", "interval"],
            max_sample_count=2,
        ),
    )

    result = _client(transport).forecast_samples(
        list(_HISTORY_ROWS),
        schema=_schema(),
        query_times=[2, 3],
        requested_columns=["target"],
        model_version=_MODEL_VERSION,
        n_samples=3,
        seed=7,
        condition=_PIN_DRIVER,
    )

    assert isinstance(result, SampleForecastResult)
    assert result.samples == (((0.0,),), ((1.0,),), ((2.0,),))
    assert result.query_times == (3,)
    assert result.diagnostics.condition_draws == 3
    assert result.plausibility == ConditionPlausibility(equality_log_density=-1.27)
    assert [payload["n_samples"] for payload in transport.payloads] == [2, 1]
    assert all(
        payload["query_mode"] == "condition"
        and payload["condition"] == _PIN_DRIVER.to_payload()
        for payload in transport.payloads
    )


def test_the_dataframe_adapter_builds_a_condition_request() -> None:
    """A pandas caller passes the block and gets the condition envelope."""
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(list(_HISTORY_ROWS))

    payload = build_forecast_payload_from_dataframe(
        frame,
        model_version=_MODEL_VERSION,
        time_index_mode="ordinal",
        query_times=[2, 3],
        target_columns=["target"],
        requested_columns=["target"],
        condition=_PIN_DRIVER,
    )

    assert payload["query_mode"] == "condition"
    assert payload["condition"] == _PIN_DRIVER.to_payload()
    assert payload["requested_columns"] == ["target"]


@pytest.mark.parametrize(
    ("field", "advertised", "message"),
    [
        ("supported_query_modes", [], "advertises nothing"),
        ("supported_condition_kinds", ["parametric"], "does not know"),
    ],
)
def test_metadata_validation_rejects_an_advertisement_this_client_cannot_serve(
    field: str, advertised: list[str], message: str
) -> None:
    """Fewer capabilities than the SDK knows are fine; none, or unknown ones, are not."""
    payload = _health_payload(
        query_modes=["forecast", "condition"],
        condition_kinds=["equality", "interval"],
    )
    payload[field] = advertised

    with pytest.raises(UnsupportedServiceContractError, match=message):
        validate_service_metadata(payload, expected_model_version=_MODEL_VERSION)
