from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pytest

from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    ForecastDiagnostics,
    ForecastOutputs,
    ForecastRequest,
    ForecastRequestMetadata,
    ForecastResponse,
    QuantileForecast,
    StructuredError,
    UnsupportedModelVersionError,
)


def test_request_models_serialize_direct_payloads_without_mutating_inputs() -> None:
    metadata = ForecastRequestMetadata(
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        return_mode="quantiles",
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="absolute_datetime",
        time_column="timestamp",
        time_scale_seconds=3600,
        use_local_normalized_time=True,
        calendar_id="custom-calendar",
        timezone="UTC",
    )
    history_rows = [
        {
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "target": 10.0,
        },
        {
            "timestamp": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "target": 11.0,
        },
    ]
    request = ForecastRequest(
        metadata=metadata,
        schema=schema,
        history_rows=history_rows,
        query_times=[
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            "2026-01-04T00:00:00Z",
        ],
        requested_columns=[0],
        n_samples=8,
        quantiles=[0.1, 0.9],
        seed=7,
    )

    assert metadata.to_payload() == {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "quantiles",
    }
    assert schema.to_payload() == {
        "time_index_mode": "absolute_datetime",
        "columns": [{"name": "target", "modality": "numeric", "role": "target"}],
        "time_column": "timestamp",
        "time_scale_seconds": 3600.0,
        "use_local_normalized_time": True,
        "calendar_id": "custom-calendar",
        "timezone": "UTC",
    }
    assert request.to_payload() == {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "quantiles",
        "time_index_mode": "absolute_datetime",
        "columns": [{"name": "target", "modality": "numeric", "role": "target"}],
        "time_column": "timestamp",
        "time_scale_seconds": 3600.0,
        "use_local_normalized_time": True,
        "calendar_id": "custom-calendar",
        "timezone": "UTC",
        "history_rows": [
            {"timestamp": "2026-01-01T00:00:00+00:00", "target": 10.0},
            {"timestamp": "2026-01-02T00:00:00+00:00", "target": 11.0},
        ],
        "query_times": [
            "2026-01-03T00:00:00+00:00",
            "2026-01-04T00:00:00+00:00",
        ],
        "requested_columns": ["target"],
        "n_samples": 8,
        "quantiles": [0.1, 0.9],
        "seed": 7,
    }
    assert isinstance(history_rows[0]["timestamp"], datetime)


def test_request_models_reject_direct_validation_edges() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        ColumnSpec(
            name="target",
            modality="numeric",
            lower_bound=2.0,
            upper_bound=1.0,
        )

    with pytest.raises(ValueError, match="schema_version"):
        ForecastRequestMetadata(
            model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
            schema_version="v2",
        )

    with pytest.raises(ValueError, match="query_mode"):
        ForecastRequestMetadata(
            model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
            query_mode=cast(Any, "complete"),
        )

    with pytest.raises(ValueError, match="quantiles may be provided only"):
        ForecastRequest(
            metadata=ForecastRequestMetadata(
                model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
                return_mode="mean",
            ),
            schema=DataFrameSchema(
                columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
                time_index_mode="ordinal",
            ),
            history_rows=[{"target": 10.0}],
            query_times=[1],
            quantiles=[0.1],
        )


def test_column_spec_serializes_one_sided_bounds() -> None:
    lower_only = ColumnSpec(
        name="price_floor",
        modality="numeric",
        role="target",
        lower_bound=0.0,
    )
    upper_only = ColumnSpec(
        name="inventory_cap",
        modality="numeric",
        role="target",
        upper_bound=100.0,
    )

    assert lower_only.to_payload() == {
        "name": "price_floor",
        "modality": "numeric",
        "role": "target",
        "lower_bound": 0.0,
    }
    assert upper_only.to_payload() == {
        "name": "inventory_cap",
        "modality": "numeric",
        "role": "target",
        "upper_bound": 100.0,
    }


def test_response_helper_models_parse_direct_payloads() -> None:
    error = StructuredError.from_payload(
        {
            "code": "VALIDATION_ERROR",
            "message": "bad request",
            "field": "query_times",
        }
    )
    diagnostics = ForecastDiagnostics.from_payload(
        {"history_rows": 2, "horizon_count": 1, "seed": 7}
    )
    mean_outputs = ForecastOutputs.from_payload(
        {
            "query_times": [1],
            "requested_columns": ["target"],
            "mean": [[12.0]],
            "samples": None,
            "quantiles": None,
        },
        return_mode="mean",
        expected_horizon_count=1,
        expected_requested_column_count=1,
    )
    sample_outputs = ForecastOutputs.from_payload(
        {
            "query_times": [1],
            "requested_columns": ["target"],
            "mean": None,
            "samples": [[[12.0]], [[12.5]]],
            "quantiles": None,
        },
        return_mode="samples",
        expected_horizon_count=1,
        expected_requested_column_count=1,
        expected_sample_count=2,
    )
    quantile_outputs = ForecastOutputs.from_payload(
        {
            "query_times": [1],
            "requested_columns": ["target"],
            "mean": None,
            "samples": None,
            "quantiles": [
                {"quantile": 0.1, "values": [[11.0]]},
                {"quantile": 0.9, "values": [[13.0]]},
            ],
        },
        return_mode="quantiles",
        expected_horizon_count=1,
        expected_requested_column_count=1,
        expected_quantiles=[0.1, 0.9],
    )

    assert error == StructuredError(
        code="VALIDATION_ERROR",
        message="bad request",
        field="query_times",
    )
    assert diagnostics.history_rows == 2
    assert diagnostics.horizon_count == 1
    assert diagnostics.seed == 7
    assert mean_outputs.mean == ((12.0,),)
    assert sample_outputs.samples == (((12.0,),), ((12.5,),))
    assert isinstance(quantile_outputs.quantiles, tuple)
    assert isinstance(quantile_outputs.quantiles[0], QuantileForecast)
    np.testing.assert_allclose(
        quantile_outputs.quantiles[0].to_numpy(),
        np.array([[11.0]]),
    )


def test_response_models_reject_direct_validation_edges() -> None:
    with pytest.raises(ValueError, match="errors.field"):
        StructuredError.from_payload(
            {"code": "VALIDATION_ERROR", "message": "bad request", "field": 1}
        )

    with pytest.raises(ValueError, match="diagnostics.history_rows"):
        ForecastDiagnostics.from_payload({"history_rows": 0, "horizon_count": 1})

    with pytest.raises(ValueError, match="outputs.samples must be null"):
        ForecastOutputs.from_payload(
            {
                "query_times": [1],
                "requested_columns": ["target"],
                "mean": [[12.0]],
                "samples": [[[12.0]]],
                "quantiles": None,
            },
            return_mode="mean",
            expected_horizon_count=1,
            expected_requested_column_count=1,
        )

    with pytest.raises(ValueError, match="forecast response quantiles mismatch"):
        ForecastOutputs.from_payload(
            {
                "query_times": [1],
                "requested_columns": ["target"],
                "mean": None,
                "samples": None,
                "quantiles": [
                    {"quantile": 0.2, "values": [[11.0]]},
                    {"quantile": 0.9, "values": [[13.0]]},
                ],
            },
            return_mode="quantiles",
            expected_horizon_count=1,
            expected_requested_column_count=1,
            expected_quantiles=[0.1, 0.9],
        )


def test_forecast_response_rejects_request_scoped_metadata_mismatches() -> None:
    request_payload = {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "mean",
        "query_times": [1],
        "requested_columns": ["target"],
    }

    mismatched_model_version = _mean_response_payload()
    mismatched_model_version["model_version"] = "jointfm-inference:9.9.9+ckpt.other"
    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        ForecastResponse.from_payload(
            mismatched_model_version,
            request_payload=request_payload,
        )

    mismatched_return_mode = _sample_response_payload()
    with pytest.raises(ValueError, match="return_mode mismatch"):
        ForecastResponse.from_payload(
            mismatched_return_mode,
            request_payload=request_payload,
        )

    mismatched_query_times = _mean_response_payload()
    mismatched_query_times["outputs"]["query_times"] = [2]
    with pytest.raises(ValueError, match="query_times mismatch"):
        ForecastResponse.from_payload(
            mismatched_query_times,
            request_payload=request_payload,
        )

    mismatched_requested_columns = _mean_response_payload()
    mismatched_requested_columns["outputs"]["requested_columns"] = ["other"]
    with pytest.raises(ValueError, match="requested_columns mismatch"):
        ForecastResponse.from_payload(
            mismatched_requested_columns,
            request_payload=request_payload,
        )

    mismatched_diagnostics = _mean_response_payload()
    mismatched_diagnostics["diagnostics"]["horizon_count"] = 2
    with pytest.raises(ValueError, match="diagnostics.horizon_count mismatch"):
        ForecastResponse.from_payload(
            mismatched_diagnostics,
            request_payload=request_payload,
        )


def test_forecast_response_rejects_sample_bound_violations() -> None:
    request_payload = {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "samples",
        "time_index_mode": "ordinal",
        "columns": [
            {
                "name": "target",
                "modality": "numeric",
                "role": "target",
                "lower_bound": 0.0,
                "upper_bound": 20.0,
            }
        ],
        "query_times": [1],
        "requested_columns": ["target"],
        "n_samples": 1,
    }

    valid_response = _sample_response_payload()
    result = ForecastResponse.from_payload(valid_response, request_payload=request_payload)
    assert result.outputs.samples == (((12.0,),),)

    below_lower_bound = _sample_response_payload()
    below_lower_bound["outputs"]["samples"] = [[[-1.0]]]
    with pytest.raises(ValueError, match="violates requested lower_bound"):
        ForecastResponse.from_payload(below_lower_bound, request_payload=request_payload)

    above_upper_bound = _sample_response_payload()
    above_upper_bound["outputs"]["samples"] = [[[21.0]]]
    with pytest.raises(ValueError, match="violates requested upper_bound"):
        ForecastResponse.from_payload(above_upper_bound, request_payload=request_payload)


def _mean_response_payload() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "checkpoint_version": "smoke-1",
        "head": "dummy",
        "query_mode": "forecast",
        "return_mode": "mean",
        "outputs": {
            "query_times": [1],
            "requested_columns": ["target"],
            "mean": [[12.0]],
            "samples": None,
            "quantiles": None,
        },
        "diagnostics": {"history_rows": 2, "horizon_count": 1, "seed": 7},
        "errors": [],
    }


def _sample_response_payload() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "checkpoint_version": "smoke-1",
        "head": "dummy",
        "query_mode": "forecast",
        "return_mode": "samples",
        "outputs": {
            "query_times": [1],
            "requested_columns": ["target"],
            "mean": None,
            "samples": [[[12.0]]],
            "quantiles": None,
        },
        "diagnostics": {"history_rows": 2, "horizon_count": 1, "seed": 7},
        "errors": [],
    }