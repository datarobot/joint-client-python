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

"""Tests for the ``log_prob`` return mode on the client side.

Scoring inverts the direction of every other return mode: the caller supplies
the values and the deployment answers how plausible they are. That shows up in
three places, and the split below follows them. The request carries observed
rows nothing else carries, and a projection narrower than the scored joint asks
a different question than the caller believes they asked. The response carries
a block with no column axis whose summary fields only repeat what ``values``
already says. Under a condition the two meet: a pinned column is projected and
supplied like any other, because the service refuses a row contradicting the
pin rather than scoring it, while the score itself stays the conditional
density of the columns the request does not condition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

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
    JointFMClient,
    LogProbResult,
    build_forecast_payload_from_dataframe,
)
from jointfm_client.contract import ReturnMode

_MODEL_VERSION = "jointfm-inference:0.3.0+ckpt.sdk-test"
_HISTORY_ROWS = (
    {"driver": 1.0, "target": 10.0},
    {"driver": 1.1, "target": 11.0},
)
_QUERY_ROWS = (
    {"driver": 1.2, "target": 12.0},
    {"driver": 1.3, "target": 13.0},
)
_PINNED_QUERY_ROWS = (
    {"driver": 1.2, "target": 12.0},
    {"driver": 1.5, "target": 13.0},
)
_PIN_DRIVER = EqualityCondition(column="driver", value=1.5, query_time_indices=[1])


def _schema() -> DataFrameSchema:
    """Build one two-column ordinal schema for request-level tests."""
    return DataFrameSchema(
        columns=(
            ColumnSpec(name="driver", modality="numeric"),
            ColumnSpec(name="target", modality="numeric", role="target"),
        ),
        time_index_mode="ordinal",
    )


def _request(
    *,
    return_mode: ReturnMode = "log_prob",
    query_rows: Any = _QUERY_ROWS,
    requested_columns: tuple[str, ...] | None = ("driver", "target"),
    condition: Condition | None = None,
) -> ForecastRequest:
    """Build one scoring request against the two-column schema."""
    return ForecastRequest(
        metadata=ForecastRequestMetadata(
            model_version=_MODEL_VERSION,
            query_mode="forecast" if condition is None else "condition",
            return_mode=return_mode,
        ),
        schema=_schema(),
        history_rows=_HISTORY_ROWS,
        query_times=(2, 3),
        requested_columns=requested_columns,
        condition=condition,
        query_rows=query_rows,
        seed=7,
    )


class _ScoringTransport:
    """Fake JSON transport for a deployment that serves one canned score."""

    def __init__(
        self,
        *,
        health_payload: Mapping[str, Any],
        predict_payload: Mapping[str, Any],
    ) -> None:
        """Remember the advertisement and the canned score to serve."""
        self.health_payload = health_payload
        self.predict_payload = predict_payload
        self.payloads: list[dict[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Serve the health advertisement on the local health route."""
        assert url == "http://127.0.0.1:8080/healthz"
        return self.health_payload

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record the scoring payload and answer it."""
        assert url == "http://127.0.0.1:8080/predict"
        self.payloads.append(dict(payload))
        return self.predict_payload


def _client(transport: _ScoringTransport) -> JointFMClient:
    """Build a local-service client over the fake transport."""
    return JointFMClient(
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        transport=transport,
    )


def test_scoring_needs_the_values_it_scores() -> None:
    """The mode and the observed rows are one request, not two independent options."""
    with pytest.raises(ValueError, match="go together"):
        _request(query_rows=None)

    with pytest.raises(ValueError, match="go together"):
        _request(return_mode="mean", requested_columns=("target",))


def test_every_future_position_needs_its_observed_row() -> None:
    """One score exists per query time, so a missing row has no answer to give."""
    with pytest.raises(ValueError, match="one row per query time"):
        _request(query_rows=(_QUERY_ROWS[0],))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"driver": 1.3}, "missing declared columns"),
        ({"driver": 1.3, "target": None}, "must carry an observed value"),
        ({"driver": 1.3, "target": float("nan")}, "must be finite"),
    ],
)
def test_a_row_that_cannot_be_scored_is_refused_before_any_round_trip(
    row: Mapping[str, Any],
    message: str,
) -> None:
    """The score is joint over every declared column, so each one needs a value."""
    with pytest.raises(ValueError, match=message):
        _request(query_rows=(_QUERY_ROWS[0], row))


def test_a_partial_projection_is_refused_because_it_scores_something_else() -> None:
    """Dropping a column would score a narrower joint than the caller believes."""
    with pytest.raises(ValueError, match="every declared column in declared order"):
        _request(requested_columns=("target",))


def test_a_score_may_leave_out_no_column_at_all_under_a_condition() -> None:
    """A condition excuses nothing: the scorer covers the whole declared joint.

    Omitting the projection is the natural spelling, because the service's own
    default already resolves to every declared column in declared order.
    """
    stated = _request(
        requested_columns=("driver", "target"),
        query_rows=_PINNED_QUERY_ROWS,
        condition=_PIN_DRIVER,
    )
    assert stated.to_payload()["requested_columns"] == ["driver", "target"]

    default_projection = _request(
        requested_columns=None,
        query_rows=_PINNED_QUERY_ROWS,
        condition=_PIN_DRIVER,
    )
    assert "requested_columns" not in default_projection.to_payload()

    with pytest.raises(ValueError, match="every declared column in declared order"):
        _request(
            requested_columns=("target",),
            query_rows=_PINNED_QUERY_ROWS,
            condition=_PIN_DRIVER,
        )


@pytest.mark.parametrize(
    ("fixture_name", "condition", "requested_columns", "query_rows"),
    [
        ("forecast_log_prob_request", None, ("driver", "target"), _QUERY_ROWS),
        (
            "condition_log_prob_request",
            _PIN_DRIVER,
            ("driver", "target"),
            _PINNED_QUERY_ROWS,
        ),
    ],
)
def test_the_payload_carries_the_rows_the_service_scores(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    fixture_name: str,
    condition: Condition | None,
    requested_columns: tuple[str, ...],
    query_rows: tuple[Mapping[str, Any], ...],
) -> None:
    """What the request builds is what the checked-in service fixture contains."""
    payload = _request(
        requested_columns=requested_columns,
        condition=condition,
        query_rows=query_rows,
    ).to_payload()

    assert payload == json_fixture_loader(fixture_name)


def test_the_scored_response_parses_into_its_own_result(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A score has a horizon axis and no column axis, and says so in its result."""
    result = ForecastResponse.from_payload(
        json_fixture_loader("forecast_log_prob_response"),
        request_payload=json_fixture_loader("forecast_log_prob_request"),
    )

    assert isinstance(result, LogProbResult)
    assert result.log_prob.values == (-2.5, -3.25)
    assert result.log_prob.nll_values == (2.5, 3.25)
    assert result.log_prob.total == pytest.approx(-5.75)
    assert result.log_prob.mean == pytest.approx(-2.875)
    assert result.plausibility is None
    assert result.outputs.log_prob is result.log_prob


def test_the_scored_conversions_keep_the_horizon_axis(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Tidy carries one row per scored position; wide carries the request summary."""
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    result = ForecastResponse.from_payload(
        json_fixture_loader("forecast_log_prob_response"),
        request_payload=json_fixture_loader("forecast_log_prob_request"),
    )
    assert isinstance(result, LogProbResult)

    assert result.to_numpy().tolist() == [-2.5, -3.25]
    tidy = result.to_pandas_tidy()
    assert list(tidy.columns) == ["query_time", "log_prob", "nll"]
    assert tidy["query_time"].tolist() == [2, 3]
    wide = result.to_pandas_wide()
    assert list(wide.columns) == ["total", "mean", "nll_total", "nll_mean"]
    assert len(wide) == 1


@pytest.mark.parametrize("field", ["total", "mean", "nll_total", "nll_mean"])
def test_a_summary_that_disagrees_with_its_scores_is_refused(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    field: str,
) -> None:
    """The summaries are redundant, so a payload that disagrees with itself is broken."""
    payload = json_fixture_loader("forecast_log_prob_response")
    payload["outputs"]["log_prob"][field] += 1.0

    with pytest.raises(ValueError, match=f"outputs.log_prob.{field} disagrees"):
        ForecastResponse.from_payload(payload)


def test_a_scored_response_is_never_read_as_another_mode(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A mode whose block is absent must fail on its own field, not on another's."""
    payload = json_fixture_loader("forecast_mean_response")
    payload["return_mode"] = "log_prob"

    with pytest.raises(ValueError, match="outputs.log_prob"):
        ForecastResponse.from_payload(payload)


def test_the_client_scores_observed_rows_and_types_the_answer(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """The typed helper puts the rows on the wire and types what comes back."""
    transport = _ScoringTransport(
        health_payload=json_fixture_loader("health_metadata"),
        predict_payload=json_fixture_loader("forecast_log_prob_response"),
    )

    result = _client(transport).forecast_log_prob(
        list(_HISTORY_ROWS),
        schema=_schema(),
        query_times=[2, 3],
        query_rows=list(_QUERY_ROWS),
        requested_columns=["driver", "target"],
        model_version=_MODEL_VERSION,
        seed=7,
    )

    assert isinstance(result, LogProbResult)
    assert result.log_prob.values == (-2.5, -3.25)
    sent = transport.payloads[0]
    assert sent["return_mode"] == "log_prob"
    assert sent["query_rows"] == [dict(row) for row in _QUERY_ROWS]


def test_the_client_scores_under_a_condition(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A conditioned score answers every position and reports its pin."""
    health_payload = json_fixture_loader("health_metadata")
    health_payload["supported_query_modes"] = ["forecast", "condition"]
    health_payload["supported_condition_kinds"] = ["equality", "interval"]
    transport = _ScoringTransport(
        health_payload=health_payload,
        predict_payload=json_fixture_loader("condition_log_prob_response"),
    )

    result = _client(transport).forecast_log_prob(
        list(_HISTORY_ROWS),
        schema=_schema(),
        query_times=[2, 3],
        query_rows=list(_PINNED_QUERY_ROWS),
        requested_columns=["driver", "target"],
        model_version=_MODEL_VERSION,
        seed=7,
        condition=_PIN_DRIVER,
    )

    assert isinstance(result, LogProbResult)
    assert result.query_times == (2, 3)
    assert result.log_prob.values == (-2.5, -1.75)
    assert result.plausibility == ConditionPlausibility(equality_log_density=-1.27)
    sent = transport.payloads[0]
    assert sent["query_mode"] == "condition"
    assert sent["query_rows"] == [dict(row) for row in _PINNED_QUERY_ROWS]


def test_the_dataframe_adapter_encodes_observed_rows_like_history() -> None:
    """A pandas caller gets both arrays through the same column encoding."""
    pandas = pytest.importorskip("pandas")

    payload = build_forecast_payload_from_dataframe(
        pandas.DataFrame(list(_HISTORY_ROWS)),
        model_version=_MODEL_VERSION,
        time_index_mode="ordinal",
        query_times=[2, 3],
        target_columns=["target"],
        requested_columns=["driver", "target"],
        return_mode="log_prob",
        query_rows=pandas.DataFrame(list(_QUERY_ROWS)),
    )

    assert payload["return_mode"] == "log_prob"
    assert payload["query_rows"] == [dict(row) for row in _QUERY_ROWS]
