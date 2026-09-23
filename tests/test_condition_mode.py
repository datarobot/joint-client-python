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
validate what is wrong independently of any deployment — including two
conditions on one column at a shared position — so a caller learns it without a
round trip. The capability gate refuses what *this* deployment has
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
    Condition,
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
    resolve_conditions,
    validate_service_metadata,
)
from jointfm_client.contract import QueryMode, condition_kinds
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
    condition: Condition | list[Condition] | None,
    *,
    requested_columns: tuple[str, ...] | None = ("target",),
    query_mode: QueryMode = "condition",
) -> ForecastRequest:
    """Build one condition request against the three-column schema."""
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
        condition=condition,
    )


def _health(
    *,
    query_modes: tuple[str, ...] = ("forecast", "condition"),
    condition_kinds: tuple[str, ...] = ("equality", "interval"),
) -> HealthMetadata:
    """Build one advertisement without going through a transport."""
    return HealthMetadata(
        status="ok",
        schema_version="v5",
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


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "more than once"),
        ([-1], "must not be negative"),
        ([True], "must hold integers"),
        ("0", "JSON array"),
    ],
    ids=["empty", "duplicate", "negative", "bool", "string"],
)
def test_a_condition_rejects_positions_that_name_nothing_usable(
    positions: Any, message: str
) -> None:
    """``None`` covers every position; an explicit list must name real, distinct ones."""
    with pytest.raises(ValueError, match=message):
        EqualityCondition(column="driver", value=1.0, query_time_indices=positions)


def test_a_condition_keeps_its_positions_as_an_immutable_tuple() -> None:
    """A frozen condition must not change because the caller's list did."""
    positions = [0, 2]
    condition = EqualityCondition(
        column="driver", value=1.0, query_time_indices=positions
    )
    positions.append(1)

    assert condition.query_time_indices == (0, 2)


def test_one_condition_and_a_list_of_them_resolve_alike() -> None:
    """A single condition is the one-element list, so callers need not wrap it."""
    pin = EqualityCondition(column="driver", value=1.0)

    assert resolve_conditions(pin) == (pin,)
    assert resolve_conditions([pin]) == (pin,)


@pytest.mark.parametrize(
    ("first", "second", "overlap"),
    [
        ((0,), (0, 1), r"query_time_indices \[0\]"),
        (None, (1,), "every position"),
        (None, None, "every position"),
    ],
    ids=["explicit", "null_against_explicit", "null_against_null"],
)
def test_two_conditions_on_one_column_at_a_shared_position_are_refused(
    first: tuple[int, ...] | None, second: tuple[int, ...] | None, overlap: str
) -> None:
    """A column carries one condition per position, whatever the two kinds are."""
    with pytest.raises(
        ValueError,
        match=rf"'driver' carries more than one condition at {overlap}: "
        r"condition\[1\] overlaps condition\[0\]",
    ):
        resolve_conditions(
            [
                EqualityCondition(column="driver", value=1.0, query_time_indices=first),
                IntervalCondition(
                    column="driver", lower=0.0, upper=2.0, query_time_indices=second
                ),
            ]
        )


def test_one_column_may_carry_conditions_at_disjoint_positions() -> None:
    """Disjoint positions never meet, so each keeps its own condition."""
    conditions = resolve_conditions(
        [
            EqualityCondition(column="driver", value=1.0, query_time_indices=[0]),
            EqualityCondition(column="driver", value=2.0, query_time_indices=[1]),
        ]
    )

    assert len(conditions) == 2


def test_different_columns_may_share_every_position() -> None:
    """A pin and a band on different columns meet at a position; that is the point."""
    conditions = resolve_conditions(
        [
            EqualityCondition(column="driver", value=1.0),
            IntervalCondition(column="hedge", lower=0.0, upper=None),
        ]
    )

    assert condition_kinds(conditions) == ("equality", "interval")


@pytest.mark.parametrize(
    ("condition", "message"),
    [([], "must not be empty"), (["driver"], "must be an EqualityCondition")],
    ids=["empty", "not_a_condition"],
)
def test_a_condition_list_must_hold_conditions(condition: Any, message: str) -> None:
    """An empty list conditions nothing, and only the two condition kinds exist."""
    with pytest.raises(ValueError, match=message):
        resolve_conditions(condition)


def test_the_mode_and_the_condition_must_agree() -> None:
    """One without the other means a different request than the caller wrote."""
    pin = EqualityCondition(column="driver", value=1.0)

    with pytest.raises(ValueError, match="go together"):
        _request(pin, query_mode="forecast")
    with pytest.raises(ValueError, match="go together"):
        _request(None)


@pytest.mark.parametrize(
    ("condition", "requested_columns", "message"),
    [
        (
            EqualityCondition(column="absent", value=1.0),
            ("target",),
            "undeclared columns",
        ),
        (
            EqualityCondition(column="driver", value=1.0, query_time_indices=[1, 9]),
            ("target",),
            r"query_time_indices \[9\] outside the 2 requested future positions",
        ),
        (
            [
                EqualityCondition(column="driver", value=1.0),
                EqualityCondition(column="hedge", value=0.5),
                EqualityCondition(column="target", value=2.0, query_time_indices=[1]),
            ],
            None,
            "pinning every column at query_time_indices position 1.*log_prob",
        ),
        (
            [
                EqualityCondition(column="driver", value=1.0, query_time_indices=[0]),
                EqualityCondition(column="hedge", value=0.5, query_time_indices=[0]),
                IntervalCondition(
                    column="target", lower=0.0, upper=None, query_time_indices=[0]
                ),
            ],
            None,
            "at least one column unconditioned at query_time_indices position 0",
        ),
    ],
    ids=["undeclared", "outside", "all_pinned", "all_conditioned"],
)
def test_a_request_is_checked_against_its_own_schema_before_any_round_trip(
    condition: Condition | list[Condition],
    requested_columns: tuple[str, ...] | None,
    message: str,
) -> None:
    """The caller learns which column or position it got wrong without a request."""
    with pytest.raises(ValueError, match=message):
        _request(condition, requested_columns=requested_columns)


def test_every_column_may_be_conditioned_somewhere_if_never_all_at_once() -> None:
    """The read-out rule holds per position, not over the request as a whole."""
    request = _request(
        [
            EqualityCondition(column="driver", value=1.0),
            EqualityCondition(column="hedge", value=0.5, query_time_indices=[0]),
            EqualityCondition(column="target", value=2.0, query_time_indices=[1]),
        ],
        requested_columns=None,
    )

    assert len(request.to_payload()["conditions"]) == 3


def test_an_interval_column_may_still_be_read_out() -> None:
    """An interval fixes a region, so the column's distribution inside it is an answer."""
    request = _request(
        IntervalCondition(column="driver", lower=0.5, upper=1.5),
        requested_columns=("driver", "target"),
    )

    assert request.to_payload()["requested_columns"] == ["driver", "target"]


def test_a_pinned_column_may_be_read_out_in_any_position() -> None:
    """A projection is condition-agnostic, so a pin is nameable like any column.

    Its answer is the request's own value, which is what lets a scenario frame
    be compared against an unconditioned one column for column.
    """
    request = _request(
        EqualityCondition(column="driver", value=1.0),
        requested_columns=("target", "driver"),
    )

    assert request.to_payload()["requested_columns"] == ["target", "driver"]


def test_omitting_the_projection_states_nothing_on_the_wire() -> None:
    """The default lives in the service, so the client must not invent one.

    A client-side default would be a second copy of the rule, free to drift from
    the one the deployment actually applies.
    """
    payload = _request(
        EqualityCondition(column="driver", value=1.0), requested_columns=None
    ).to_payload()

    assert "requested_columns" not in payload


def test_the_payload_carries_the_conditions_the_service_parses() -> None:
    """Each condition names its kind and its positions, ``null`` covering them all."""
    payload = build_forecast_payload(
        model_version=_MODEL_VERSION,
        schema=_schema(),
        history_rows=({"driver": 1.0, "hedge": 0.5, "target": 10.0},),
        query_times=(2, 3),
        requested_columns=("target",),
        query_mode="condition",
        condition=[
            EqualityCondition(column="driver", value=1.5),
            IntervalCondition(
                column="target", lower=None, upper=12.0, query_time_indices=[1]
            ),
        ],
    )

    assert payload["query_mode"] == "condition"
    assert "condition" not in payload
    assert payload["conditions"] == [
        {
            "column": "driver",
            "kind": "equality",
            "value": 1.5,
            "query_time_indices": None,
        },
        {
            "column": "target",
            "kind": "interval",
            "lower": None,
            "upper": 12.0,
            "query_time_indices": [1],
        },
    ]


def test_the_gate_refuses_a_deployment_that_advertises_no_condition_mode() -> None:
    """Discovery before the request is the point: the error must not cost a round trip."""
    pin = EqualityCondition(column="driver", value=1.0)

    with pytest.raises(
        UnsupportedServiceContractError, match="does not serve condition"
    ):
        require_condition_support(
            _health(query_modes=("forecast",), condition_kinds=()), pin
        )


def test_the_gate_refuses_a_kind_the_deployment_does_not_answer() -> None:
    """A deployment may serve one kind before the other; the conditions say which."""
    conditions = [
        EqualityCondition(column="driver", value=1.0),
        IntervalCondition(column="hedge", lower=0.0, upper=1.0),
    ]

    with pytest.raises(UnsupportedServiceContractError, match=r"kinds \['interval'\]"):
        require_condition_support(_health(condition_kinds=("equality",)), conditions)

    require_condition_support(_health(), conditions)


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


def test_the_response_answers_every_requested_position(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The condition covers one of two rows, and the answer still carries both."""
    request_payload = json_fixture_loader("condition_mean_request")
    assert request_payload["query_times"] == [2, 3]
    assert request_payload["conditions"][0]["query_time_indices"] == [1]

    response = ForecastResponse.from_payload(
        json_fixture_loader("condition_mean_response"),
        request_payload=request_payload,
    )

    assert response.query_times == (2, 3)
    assert response.diagnostics.horizon_count == 2


def test_a_response_narrowed_to_the_covered_position_is_refused(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A deployment answering fewer positions than asked must fail, not parse."""
    response_payload = json_fixture_loader("condition_mean_response")
    response_payload["outputs"]["query_times"] = [3]
    response_payload["outputs"]["mean"] = [[13.5]]
    response_payload["diagnostics"]["horizon_count"] = 1

    with pytest.raises(ValueError, match="query_times"):
        ForecastResponse.from_payload(
            response_payload,
            request_payload=json_fixture_loader("condition_mean_request"),
        )


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
            "schema_version": "v5",
            "image_version": "0.3.0",
            "model_version": _MODEL_VERSION,
            "checkpoint_version": "sdk-test",
            "head": "gmm",
            "query_mode": "condition",
            "return_mode": "samples",
            "outputs": {
                "query_times": [2, 3],
                "requested_columns": ["target"],
                "mean": None,
                "samples": [
                    [[-1.0], [float(start + index)]] for index in range(sample_count)
                ],
                "quantiles": None,
            },
            "plausibility": {
                "equality_log_density": -1.27,
                "region_log_probability": None,
            },
            "diagnostics": {
                "history_rows": 2,
                "horizon_count": 2,
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
        "schema_version": "v5",
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
_PIN_DRIVER = EqualityCondition(column="driver", value=1.5, query_time_indices=[1])


def test_the_client_sends_the_condition_and_reads_the_answer_back(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The typed helper carries the condition onto the wire and types what comes back."""
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
    assert result.query_times == (2, 3)
    assert result.mean == ((12.0,), (13.5,))
    assert result.plausibility == ConditionPlausibility(equality_log_density=-1.27)
    assert len(transport.payloads) == 1
    sent = transport.payloads[0]
    assert sent["query_mode"] == "condition"
    assert sent["conditions"] == [_PIN_DRIVER.to_payload()]


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
    """Every batch repeats the same conditions; the merge recounts draws, keeps plausibility."""
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
    assert result.samples == (
        ((-1.0,), (0.0,)),
        ((-1.0,), (1.0,)),
        ((-1.0,), (2.0,)),
    )
    assert result.query_times == (2, 3)
    assert result.diagnostics.condition_draws == 3
    assert result.plausibility == ConditionPlausibility(equality_log_density=-1.27)
    assert [payload["n_samples"] for payload in transport.payloads] == [2, 1]
    assert all(
        payload["query_mode"] == "condition"
        and payload["conditions"] == [_PIN_DRIVER.to_payload()]
        for payload in transport.payloads
    )


def test_the_dataframe_adapter_builds_a_condition_request() -> None:
    """A pandas caller passes the conditions and gets the condition envelope."""
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
    assert payload["conditions"] == [_PIN_DRIVER.to_payload()]
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
