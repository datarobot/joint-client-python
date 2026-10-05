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

"""Tests for the NumPy-backed forecast result values of jointfm_client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    ForecastResponse,
    JointFMClient,
    MeanForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)

_QUERY_TIME_SPELLINGS: tuple[tuple[Any, ...], ...] = (
    (5, 6, 7),
    (5.5, 6.5, 7.5),
    ("2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z", "2026-01-07T00:00:00Z"),
)
_COLUMNS = ("alpha", "beta")
_SAMPLE_COUNT = 4


def _payload(return_mode: str, outputs: dict[str, Any], query_times: Any) -> dict:
    """One forecast response over three horizons and two requested columns."""
    return {
        "schema_version": "v5",
        "image_version": "0.3.0",
        "model_version": "jointfm-inference:0.3.0+ckpt.smoke-1",
        "checkpoint_version": "smoke-1",
        "head": "dummy",
        "query_mode": "forecast",
        "return_mode": return_mode,
        "outputs": {
            "query_times": list(query_times),
            "requested_columns": list(_COLUMNS),
            "mean": None,
            "samples": None,
            "quantiles": None,
        }
        | outputs,
        "diagnostics": {
            "history_rows": 2,
            "horizon_count": len(query_times),
            "seed": 7,
        },
        "errors": [],
    }


def _samples(query_times: Any) -> SampleForecastResult:
    """Parse one sample response whose values encode their own position."""
    values = np.arange(_SAMPLE_COUNT * 3 * 2, dtype=np.float64).reshape(
        _SAMPLE_COUNT, 3, 2
    )
    result = ForecastResponse.from_payload(
        _payload("samples", {"samples": values.tolist()}, query_times)
    )
    assert isinstance(result, SampleForecastResult)
    return result


def test_parsed_values_are_read_only_float64_arrays() -> None:
    """Every forecast value block is an immutable float64 array of the response shape."""
    result = _samples(_QUERY_TIME_SPELLINGS[0])

    assert result.samples.dtype == np.float64
    assert result.samples.shape == (_SAMPLE_COUNT, 3, 2)
    assert result.to_numpy() is result.samples
    with pytest.raises(ValueError, match="read-only"):
        result.samples[0, 0, 0] = 1.0


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([[["1.0", 2.0]]], "must contain only numbers"),
        ([[[1.0, None]]], "must contain only numbers"),
        ([[[True, False]]], "must contain only numbers"),
        ([[[1.0, 2.0]], [[1.0]]], "rectangular array"),
        ([[1.0, 2.0]], "must have 3 axes"),
        ([[[1.0, 2.0, 3.0]]], "length mismatch on axis 2"),
    ],
)
def test_malformed_sample_values_are_refused(samples: Any, message: str) -> None:
    """Strings, nulls, booleans, ragged nesting, and wrong shapes never coerce."""
    payload = _payload("samples", {"samples": samples}, (5,))
    payload["outputs"]["requested_columns"] = ["alpha", "beta"]

    with pytest.raises(ValueError, match=message):
        ForecastResponse.from_payload(payload)


def test_a_non_finite_value_is_reported_by_index() -> None:
    """The first non-finite cell is named, so the defect can be located."""
    payload = _payload("samples", {"samples": [[[1.0, 2.0]]]}, (5,))
    payload["outputs"]["samples"][0][0][1] = float("inf")

    with pytest.raises(
        ValueError, match=r"outputs\.samples\[0\]\[0\]\[1\] must be finite"
    ):
        ForecastResponse.from_payload(payload)


@pytest.mark.parametrize("query_times", _QUERY_TIME_SPELLINGS)
def test_frames_match_frames_built_from_records(query_times: tuple[Any, ...]) -> None:
    """Vectorized frames keep the row order and dtypes record-built frames had."""
    result = _samples(query_times)
    values = result.samples
    tidy = pd.DataFrame.from_records(
        [
            {
                "sample": sample,
                "query_time": query_time,
                "requested_column": column,
                "value": values[sample, horizon, column_index],
            }
            for sample in range(_SAMPLE_COUNT)
            for horizon, query_time in enumerate(query_times)
            for column_index, column in enumerate(_COLUMNS)
        ]
    )
    wide = pd.DataFrame.from_records(
        [
            {"sample": sample, "query_time": query_time}
            | {
                column: values[sample, horizon, column_index]
                for column_index, column in enumerate(_COLUMNS)
            }
            for sample in range(_SAMPLE_COUNT)
            for horizon, query_time in enumerate(query_times)
        ]
    )

    pd.testing.assert_frame_equal(result.to_pandas_tidy(), tidy)
    pd.testing.assert_frame_equal(result.to_pandas_wide(), wide)


def test_mean_and_quantile_frames_follow_the_same_layout() -> None:
    """Mean frames have no leading column; quantile frames lead with the level."""
    mean_values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    mean = ForecastResponse.from_payload(
        _payload("mean", {"mean": mean_values}, (5, 6, 7))
    )
    assert isinstance(mean, MeanForecastResult)
    assert list(mean.to_pandas_tidy().columns) == [
        "query_time",
        "requested_column",
        "value",
    ]
    assert mean.to_pandas_tidy()["value"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert mean.to_pandas_wide()["beta"].tolist() == [2.0, 4.0, 6.0]

    quantiles = ForecastResponse.from_payload(
        _payload(
            "quantiles",
            {
                "quantiles": [
                    {"quantile": 0.1, "values": mean_values},
                    {"quantile": 0.9, "values": mean_values},
                ]
            },
            (5, 6, 7),
        )
    )
    assert isinstance(quantiles, QuantileForecastResult)
    assert quantiles.to_numpy().shape == (2, 3, 2)
    wide = quantiles.to_pandas_wide()
    assert wide["quantile"].tolist() == [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    assert wide["query_time"].tolist() == [5, 6, 7, 5, 6, 7]


def test_wide_frame_refuses_a_column_named_like_an_index_column() -> None:
    """A requested column called ``query_time`` would silently overwrite the index."""
    payload = _payload("mean", {"mean": [[1.0, 2.0]]}, (5,))
    payload["outputs"]["requested_columns"] = ["query_time", "beta"]
    result = ForecastResponse.from_payload(payload)

    with pytest.raises(ValueError, match="clash"):
        result.to_pandas_wide()


def test_a_callers_array_is_copied_not_frozen() -> None:
    """Parsing an in-process payload must not make the caller's buffer read-only."""
    caller_values = np.zeros((1, 1, 2))
    payload = _payload("samples", {"samples": caller_values}, (5,))

    result = ForecastResponse.from_payload(payload)

    assert isinstance(result, SampleForecastResult)
    assert caller_values.flags.writeable
    assert not result.samples.flags.writeable


def _nullable_request(*, nullable: bool) -> dict[str, Any]:
    """Request payload declaring the second requested column nullable or not."""
    return {
        "return_mode": "samples",
        "query_times": [5],
        "requested_columns": list(_COLUMNS),
        "columns": [
            {"name": "alpha", "modality": "numeric", "role": "target"},
            {
                "name": "beta",
                "modality": "numeric",
                "role": "target",
                "nullable": nullable,
            },
        ],
    }


def test_a_null_in_a_nullable_column_parses_to_nan() -> None:
    """``null`` is the observed missing state of a column declared nullable."""
    payload = _payload("samples", {"samples": [[[1.5, None]], [[2.5, 3.5]]]}, (5,))

    result = ForecastResponse.from_payload(
        payload, request_payload=_nullable_request(nullable=True)
    )

    assert isinstance(result, SampleForecastResult)
    assert result.samples.dtype == np.float64
    assert not result.samples.flags.writeable
    np.testing.assert_array_equal(result.samples, [[[1.5, np.nan]], [[2.5, 3.5]]])
    assert result.to_pandas_tidy()["value"].isna().tolist() == [
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("samples", "request_payload"),
    [
        ([[[1.5, None]]], _nullable_request(nullable=False)),
        ([[[None, 2.5]]], _nullable_request(nullable=True)),
        ([[[1.5, None]]], None),
    ],
)
def test_a_null_outside_a_nullable_column_is_refused(
    samples: Any, request_payload: dict[str, Any] | None
) -> None:
    """Without a nullable declaration for its column, ``null`` is not a value."""
    payload = _payload("samples", {"samples": samples}, (5,))

    with pytest.raises(ValueError, match=r"outputs\.samples"):
        ForecastResponse.from_payload(payload, request_payload=request_payload)


def test_nullable_columns_still_refuse_non_numbers() -> None:
    """The null-aware path accepts numbers and nulls only."""
    payload = _payload("samples", {"samples": [[[{"x": 1}, None]]]}, (5,))

    with pytest.raises(ValueError, match="only numbers and nulls"):
        ForecastResponse.from_payload(
            payload, request_payload=_nullable_request(nullable=True)
        )


class _NullDrawingTransport:
    """Answer health, and every forecast with a null draw in column ``beta``."""

    def __init__(self) -> None:
        self.forecast_payloads: list[Mapping[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Answer the health probe of a client configured without settings."""
        del url
        return _HEALTH_PAYLOAD

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        del url
        self.forecast_payloads.append(payload)
        response = _payload("samples", {"samples": [[[1.5, None]], [[2.5, 3.5]]]}, (2,))
        response["model_version"] = _HEALTH_PAYLOAD["model_version"]
        return response


_HEALTH_PAYLOAD: dict[str, Any] = {
    "status": "ok",
    "schema_version": "v5",
    "image_version": "0.3.0",
    "model_version": "jointfm-inference:0.3.0+ckpt.smoke-1",
    "checkpoint_version": "smoke-1",
    "checkpoint_path": "/models/jointfm.pt",
    "device": "cpu",
    "head": "studentt",
    "decoding_strategy": "parallel_dense",
    "supported_query_modes": ["forecast"],
    "supported_condition_kinds": [],
    "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
    "supported_time_index_modes": ["ordinal", "continuous_float", "absolute_datetime"],
    "time_index_encoding": "legacy_discrete_grid",
    "max_sample_count": 64,
}


def test_the_client_reads_a_nullable_columns_null_as_nan() -> None:
    """A declared nullable column travels out and its missing draws come back as NaN."""
    transport = _NullDrawingTransport()
    url = "https://example.com/predict"
    client = JointFMClient(health_url=url, predict_url=url, transport=transport)

    result = client.forecast_samples(
        [{"alpha": 1.0, "beta": None}, {"alpha": 2.0, "beta": 3.0}],
        schema=DataFrameSchema(
            columns=(
                ColumnSpec(name="alpha", modality="numeric", role="target"),
                ColumnSpec(
                    name="beta", modality="numeric", role="target", nullable=True
                ),
            ),
            time_index_mode="ordinal",
        ),
        query_times=[2],
        requested_columns=list(_COLUMNS),
        model_version=_HEALTH_PAYLOAD["model_version"],
        n_samples=2,
        seed=7,
    )

    [sent] = transport.forecast_payloads
    assert [column.get("nullable", False) for column in sent["columns"]] == [
        False,
        True,
    ]
    assert sent["history_rows"][0]["beta"] is None
    np.testing.assert_array_equal(result.samples, [[[1.5, np.nan]], [[2.5, 3.5]]])
