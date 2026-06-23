from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from jointfm_client import (
    ColumnSpec,
    DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE,
    DEFAULT_CALENDAR_ID,
    DataFrameSchema,
    DISTRIBUTION_NAME,
    FIRST_SUPPORTED_PYTHON_VERSION,
    ForecastRequest,
    ForecastRequestMetadata,
    ForecastResponse,
    HealthMetadata,
    IMPORT_NAMESPACE,
    JointFMServiceError,
    JointFMClient,
    JointFMConfigurationError,
    LOCAL_HEALTH_ROUTE,
    MeanForecastResult,
    QuantileForecastResult,
    LOCAL_PREDICT_ROUTE,
    SCHEMA_VERSION,
    SampleForecastResult,
    SUPPORTED_COLUMN_MODALITIES,
    SUPPORTED_COLUMN_ROLES,
    STRUCTURED_ERROR_CODES,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_RETURN_MODES,
    SUPPORTED_TIME_INDEX_MODES,
    SUPPORTED_TIME_VALUE_KINDS,
    UnsupportedModelVersionError,
    UnsupportedSchemaVersionError,
    UnsupportedServiceContractError,
    arrays_to_history_rows,
    build_continuous_query_times,
    build_datetime_query_times,
    build_forecast_payload,
    build_forecast_payload_from_arrays,
    build_forecast_payload_from_dataframe,
    build_ordinal_query_times,
    dataframe_to_history_rows,
    infer_column_specs_from_dataframe,
    validate_forecast_horizon,
    validate_service_metadata,
)


def _health_metadata() -> dict[str, object]:
    return {
        "status": "ok",
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "checkpoint_version": "smoke-1",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "dummy",
        "supported_query_modes": ["forecast"],
        "supported_return_modes": ["mean", "quantiles", "samples", "log_prob"],
        "supported_time_index_modes": [
            "absolute_datetime",
            "continuous_float",
            "ordinal",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "default_sample_count": 256,
        "max_sample_count": 4096,
        "data_generation": {
            "sampler_type": "studentt",
            "min_features": 0,
            "max_features": 12,
            "min_targets": 1,
            "max_targets": 4,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    }


def test_package_identity_contract() -> None:
    assert DISTRIBUTION_NAME == "jointfm-client"
    assert IMPORT_NAMESPACE == "jointfm_client"
    assert FIRST_SUPPORTED_PYTHON_VERSION == "3.13"
    assert SCHEMA_VERSION == "v1"


def test_service_route_contract() -> None:
    assert DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE == (
        "deployments/{deployment_id}/predictionsUnstructured"
    )
    assert LOCAL_HEALTH_ROUTE == "/healthz"
    assert LOCAL_PREDICT_ROUTE == "/predict"


def test_mode_and_error_contract() -> None:
    assert SUPPORTED_QUERY_MODES == ("forecast",)
    assert SUPPORTED_RETURN_MODES == ("mean", "samples", "quantiles", "log_prob")
    assert SUPPORTED_TIME_INDEX_MODES == (
        "ordinal",
        "continuous_float",
        "absolute_datetime",
    )
    assert SUPPORTED_COLUMN_MODALITIES == (
        "numeric",
        "categorical",
        "ordinal",
        "binary",
        "count",
        "time_value",
    )
    assert SUPPORTED_COLUMN_ROLES == (
        "target",
        "known_dynamic",
        "past_dynamic",
        "static",
        "feature",
    )
    assert SUPPORTED_TIME_VALUE_KINDS == ("continuous_float", "absolute_datetime")
    assert DEFAULT_CALENDAR_ID == "pandas-default"
    assert "SCHEMA_VERSION_MISMATCH" in STRUCTURED_ERROR_CODES
    assert "MODEL_VERSION_MISMATCH" in STRUCTURED_ERROR_CODES
    assert "INPUT_SIZE_EXCEEDED" in STRUCTURED_ERROR_CODES


def test_column_spec_serializes_supported_metadata() -> None:
    categorical_column = ColumnSpec(
        name="segment",
        modality="categorical",
        role="target",
        nullable=True,
        vocabulary_size=5,
        level_count=4,
        mapping={"low": 0, "high": 1},
        lower_bound=0,
        upper_bound=10.5,
    )
    time_column = ColumnSpec(
        name="event_time",
        modality="time_value",
        time_value_kind="absolute_datetime",
        time_value_scale_seconds=60,
        time_value_use_local_normalized_time=True,
        time_value_calendar_id="custom-calendar",
        time_value_timezone="UTC",
    )

    assert categorical_column.to_payload() == {
        "name": "segment",
        "modality": "categorical",
        "role": "target",
        "nullable": True,
        "vocabulary_size": 5,
        "level_count": 4,
        "mapping": {"low": 0, "high": 1},
        "lower_bound": 0.0,
        "upper_bound": 10.5,
    }
    assert time_column.to_payload() == {
        "name": "event_time",
        "modality": "time_value",
        "time_value_kind": "absolute_datetime",
        "time_value_scale_seconds": 60.0,
        "time_value_use_local_normalized_time": True,
        "time_value_calendar_id": "custom-calendar",
        "time_value_timezone": "UTC",
    }


def test_column_spec_rejects_invalid_time_value_metadata() -> None:
    with pytest.raises(ValueError, match="time_value options"):
        ColumnSpec(
            name="feature",
            modality="numeric",
            time_value_kind="absolute_datetime",
        )

    with pytest.raises(ValueError, match="time_value_kind='absolute_datetime'"):
        ColumnSpec(
            name="event_age",
            modality="time_value",
            time_value_kind="continuous_float",
            time_value_scale_seconds=60,
        )


def test_forecast_payload_matches_service_contract_without_mutating_inputs() -> None:
    timestamp = datetime(2026, 1, 5, tzinfo=timezone.utc)
    history_rows = [
        {
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "known_feature": 1.0,
            "target": 10.0,
        }
    ]
    schema = DataFrameSchema(
        columns=(
            ColumnSpec(name="known_feature", modality="numeric"),
            ColumnSpec(name="target", modality="numeric", role="target"),
        ),
        time_index_mode="absolute_datetime",
        time_column="timestamp",
        time_scale_seconds=86_400,
        use_local_normalized_time=True,
        calendar_id="gregorian-utc",
        timezone="UTC",
    )

    payload = build_forecast_payload(
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        schema=schema,
        history_rows=history_rows,
        query_times=[timestamp, "2026-01-06T00:00:00Z"],
        requested_columns=[1],
        return_mode="quantiles",
        n_samples=8,
        quantiles=[0.1, 0.9],
        seed=7,
    )

    assert payload == {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "quantiles",
        "time_index_mode": "absolute_datetime",
        "columns": [
            {"name": "known_feature", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "time_column": "timestamp",
        "time_scale_seconds": 86400.0,
        "use_local_normalized_time": True,
        "calendar_id": "gregorian-utc",
        "timezone": "UTC",
        "history_rows": [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "known_feature": 1.0,
                "target": 10.0,
            }
        ],
        "query_times": [
            "2026-01-05T00:00:00+00:00",
            "2026-01-06T00:00:00+00:00",
        ],
        "requested_columns": ["target"],
        "n_samples": 8,
        "quantiles": [0.1, 0.9],
        "seed": 7,
    }
    assert isinstance(history_rows[0]["timestamp"], datetime)


def test_forecast_payload_preserves_non_datetime_query_values() -> None:
    ordinal_schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )
    continuous_schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="continuous_float",
    )
    history_rows = [{"target": 1.0}]

    ordinal_payload = build_forecast_payload(
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        schema=ordinal_schema,
        history_rows=history_rows,
        query_times=[1, 2],
    )
    continuous_payload = build_forecast_payload(
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        schema=continuous_schema,
        history_rows=history_rows,
        query_times=[1, 2.5],
    )

    assert ordinal_payload["query_times"] == [1, 2]
    assert continuous_payload["query_times"] == [1, 2.5]


def test_dataframe_payload_matches_service_forecast_request_shape() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                ],
                utc=True,
            ),
            "known_feature": [1.0, 1.1, 1.2, 1.3],
            "target": [10.0, 11.0, 12.0, 13.0],
        }
    )

    payload = build_forecast_payload_from_dataframe(
        frame,
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        time_index_mode="absolute_datetime",
        time_column="timestamp",
        columns=(
            ColumnSpec(name="known_feature", modality="numeric"),
            ColumnSpec(name="target", modality="numeric", role="target"),
        ),
        query_times=["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        requested_columns=["target"],
        seed=7,
    )

    assert payload == {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "mean",
        "time_index_mode": "absolute_datetime",
        "columns": [
            {"name": "known_feature", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "time_column": "timestamp",
        "history_rows": [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "known_feature": 1.0,
                "target": 10.0,
            },
            {
                "timestamp": "2026-01-02T00:00:00+00:00",
                "known_feature": 1.1,
                "target": 11.0,
            },
            {
                "timestamp": "2026-01-03T00:00:00+00:00",
                "known_feature": 1.2,
                "target": 12.0,
            },
            {
                "timestamp": "2026-01-04T00:00:00+00:00",
                "known_feature": 1.3,
                "target": 13.0,
            },
        ],
        "query_times": [
            "2026-01-05T00:00:00+00:00",
            "2026-01-06T00:00:00+00:00",
        ],
        "requested_columns": ["target"],
        "seed": 7,
    }
    history_rows = cast(list[dict[str, object]], payload["history_rows"])
    assert list(history_rows[0]) == ["timestamp", "known_feature", "target"]


def test_dataframe_inference_handles_roles_mappings_bounds_and_time_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                utc=True,
            ),
            "segment": ["low", "high"],
            "priority": ["medium", "high"],
            "event_time": pd.to_datetime(
                ["2026-01-01T01:00:00Z", "2026-01-02T01:00:00Z"],
                utc=True,
            ),
            "target": [10.0, None],
        }
    )

    columns = infer_column_specs_from_dataframe(
        frame,
        time_column="timestamp",
        target_columns=["target"],
        categorical_columns=["segment"],
        ordinal_mappings={"priority": {"low": 0, "medium": 1, "high": 2}},
        time_value_columns=cast(Any, {"event_time": "absolute_datetime"}),
        nullable_columns=["target"],
        bounds={"target": (0.0, 100.0)},
    )

    assert [column.name for column in columns] == [
        "segment",
        "priority",
        "event_time",
        "target",
    ]
    assert columns[0].modality == "categorical"
    assert columns[0].mapping == {"high": 0, "low": 1}
    assert columns[1].modality == "ordinal"
    assert columns[1].level_count == 3
    assert columns[2].modality == "time_value"
    assert columns[2].time_value_kind == "absolute_datetime"
    assert columns[3].role == "target"
    assert columns[3].nullable is True
    assert columns[3].lower_bound == 0.0
    assert columns[3].upper_bound == 100.0

    rows = dataframe_to_history_rows(
        frame,
        DataFrameSchema(
            columns=columns,
            time_index_mode="absolute_datetime",
            time_column="timestamp",
        ),
    )

    assert rows[0]["segment"] == 1
    assert rows[0]["priority"] == 1
    assert rows[0]["event_time"] == "2026-01-01T01:00:00+00:00"
    assert rows[1]["target"] is None


def test_array_payload_builds_ordered_rows_with_metadata() -> None:
    columns = (
        ColumnSpec(name="target", modality="numeric", role="target"),
        ColumnSpec(
            name="segment",
            modality="categorical",
            mapping={"low": 0, "high": 1},
            vocabulary_size=2,
        ),
    )

    payload = build_forecast_payload_from_arrays(
        np.array([[10.0, "low"], [11.0, "high"]], dtype=object),
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
        columns=columns,
        time_index_mode="ordinal",
        query_times=[2, 3],
        requested_columns=[0],
    )

    assert payload["columns"] == [
        {"name": "target", "modality": "numeric", "role": "target"},
        {
            "name": "segment",
            "modality": "categorical",
            "vocabulary_size": 2,
            "mapping": {"low": 0, "high": 1},
        },
    ]
    assert payload["history_rows"] == [
        {"target": 10.0, "segment": 0},
        {"target": 11.0, "segment": 1},
    ]
    assert payload["requested_columns"] == ["target"]


def test_arrays_to_history_rows_requires_time_values_with_time_column() -> None:
    with pytest.raises(ValueError, match="provided together"):
        arrays_to_history_rows(
            [[1.0]],
            columns=(ColumnSpec(name="target", modality="numeric"),),
            time_column="timestamp",
        )


def test_query_time_builders_and_horizon_validation() -> None:
    assert build_datetime_query_times(
        ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        periods=2,
    ) == ["2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"]
    assert build_ordinal_query_times([0, 1, 2], periods=2) == [3, 4]
    assert build_continuous_query_times([0.0, 0.5, 1.0], periods=2) == [1.5, 2.0]
    assert validate_forecast_horizon(
        ["2026-01-01T00:00:00Z"],
        ["2026-01-02T00:00:00Z"],
        time_index_mode="absolute_datetime",
    ) == ["2026-01-02T00:00:00+00:00"]

    with pytest.raises(ValueError, match="strictly after"):
        validate_forecast_horizon([0, 1, 2], [2], time_index_mode="ordinal")

    with pytest.raises(ValueError, match="strictly monotonically"):
        validate_forecast_horizon(
            [0.0, 1.0],
            [2.0, 2.0],
            time_index_mode="continuous_float",
        )


def test_dataframe_adapter_rejects_invalid_frames_and_column_options() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        infer_column_specs_from_dataframe([], time_column="timestamp")

    with pytest.raises(ValueError, match="at least one row"):
        infer_column_specs_from_dataframe(pd.DataFrame({"target": []}))

    with pytest.raises(ValueError, match="time_column"):
        infer_column_specs_from_dataframe(pd.DataFrame({"target": [1.0]}), time_column="timestamp")

    with pytest.raises(ValueError, match="non-time column"):
        infer_column_specs_from_dataframe(
            pd.DataFrame({"timestamp": [0]}),
            time_column="timestamp",
        )

    with pytest.raises(ValueError, match="unknown columns"):
        infer_column_specs_from_dataframe(
            pd.DataFrame({"target": [1.0]}),
            categorical_columns=["segment"],
        )

    with pytest.raises(ValueError, match="conflicting roles"):
        infer_column_specs_from_dataframe(
            pd.DataFrame({"target": [1.0]}),
            target_columns=["target"],
            feature_columns=["target"],
        )


def test_dataframe_adapter_infers_extra_modalities_and_roles() -> None:
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0],
            "is_open": [True, False],
            "volume": [10, 20],
            "elapsed": [0.5, 1.5],
        }
    )

    columns = infer_column_specs_from_dataframe(
        frame,
        target_columns=["target"],
        known_dynamic_columns=["is_open"],
        count_columns=["volume"],
        time_value_columns=["elapsed"],
    )

    assert [column.modality for column in columns] == [
        "numeric",
        "binary",
        "count",
        "time_value",
    ]
    assert columns[1].role == "known_dynamic"
    assert columns[3].time_value_kind == "continuous_float"


def test_dataframe_history_rows_reject_bad_values() -> None:
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric"),),
        time_index_mode="ordinal",
        time_column="step",
    )

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        dataframe_to_history_rows([], schema)

    with pytest.raises(ValueError, match="missing history columns"):
        dataframe_to_history_rows(pd.DataFrame({"target": [1.0]}), schema)

    with pytest.raises(ValueError, match="not nullable"):
        dataframe_to_history_rows(
            pd.DataFrame({"step": [0], "target": [None]}),
            schema,
        )

    with pytest.raises(ValueError, match="not present in column mapping"):
        dataframe_to_history_rows(
            pd.DataFrame({"segment": ["medium"]}),
            DataFrameSchema(
                columns=(
                    ColumnSpec(
                        name="segment",
                        modality="categorical",
                        mapping={"low": 0, "high": 1},
                    ),
                ),
                time_index_mode="ordinal",
            ),
        )


def test_array_adapter_rejects_invalid_shapes_and_time_values() -> None:
    columns = (ColumnSpec(name="target", modality="numeric"),)

    with pytest.raises(ValueError, match="two-dimensional"):
        arrays_to_history_rows([1.0, 2.0], columns=columns)

    with pytest.raises(ValueError, match="at least one row"):
        arrays_to_history_rows(np.empty((0, 1)), columns=columns)

    with pytest.raises(ValueError, match="column count"):
        arrays_to_history_rows([[1.0, 2.0]], columns=columns)

    with pytest.raises(ValueError, match="time_values length"):
        arrays_to_history_rows(
            [[1.0]],
            columns=columns,
            time_column="timestamp",
            time_values=[0, 1],
        )

    with pytest.raises(ValueError, match="time_values are required"):
        build_forecast_payload_from_arrays(
            [[1.0]],
            model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
            columns=columns,
            time_index_mode="continuous_float",
            query_times=[1.0],
        )


def test_query_builders_reject_invalid_inputs() -> None:
    assert build_datetime_query_times(
        ["2026-01-01T00:00:00Z"],
        periods=2,
        frequency="1D",
    ) == ["2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00"]

    with pytest.raises(ValueError, match="periods"):
        build_ordinal_query_times([0], periods=0)

    with pytest.raises(ValueError, match="frequency is required"):
        build_datetime_query_times(["2026-01-01T00:00:00Z"], periods=1)

    with pytest.raises(ValueError, match="positive integer"):
        build_ordinal_query_times([0], periods=1, step=0)

    with pytest.raises(ValueError, match="positive finite"):
        build_continuous_query_times([0.0, 1.0], periods=1, step=float("nan"))

    with pytest.raises(ValueError, match="must be a sequence"):
        validate_forecast_horizon("bad", [1], time_index_mode="ordinal")

    with pytest.raises(ValueError, match="must be numeric"):
        validate_forecast_horizon([0.0], ["bad"], time_index_mode="continuous_float")


def test_forecast_request_rejects_service_validation_edges() -> None:
    schema = DataFrameSchema(
        columns=(
            ColumnSpec(name="known_feature", modality="numeric"),
            ColumnSpec(name="target", modality="numeric", role="target"),
        ),
        time_index_mode="ordinal",
    )
    metadata = ForecastRequestMetadata(
        model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
    )

    with pytest.raises(ValueError, match="history_rows"):
        ForecastRequest(
            metadata=metadata,
            schema=schema,
            history_rows=[],
            query_times=[2],
        )
    with pytest.raises(ValueError, match="query_times"):
        ForecastRequest(
            metadata=metadata,
            schema=schema,
            history_rows=[{"known_feature": 1.0, "target": 10.0}],
            query_times=[],
        )
    with pytest.raises(ValueError, match="duplicates"):
        ForecastRequest(
            metadata=metadata,
            schema=schema,
            history_rows=[{"known_feature": 1.0, "target": 10.0}],
            query_times=[2],
            requested_columns=[1, "target"],
        )
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ForecastRequest(
            metadata=ForecastRequestMetadata(
                model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
                return_mode="quantiles",
            ),
            schema=schema,
            history_rows=[{"known_feature": 1.0, "target": 10.0}],
            query_times=[2],
            quantiles=[0.0],
        )
    with pytest.raises(ValueError, match="n_samples"):
        ForecastRequest(
            metadata=metadata,
            schema=schema,
            history_rows=[{"known_feature": 1.0, "target": 10.0}],
            query_times=[2],
            n_samples=0,
        )
    with pytest.raises(ValueError, match="return_mode"):
        ForecastRequestMetadata(
            model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
            return_mode=cast(Any, "median"),
        )
    with pytest.raises(ValueError, match="query_row_ids"):
        ForecastRequest(
            metadata=metadata,
            schema=schema,
            history_rows=[{"known_feature": 1.0, "target": 10.0}],
            query_times=[2],
            query_row_ids=[0],
        )


def test_dataframe_schema_rejects_absolute_datetime_without_time_column() -> None:
    with pytest.raises(ValueError, match="time_column"):
        DataFrameSchema(
            columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
            time_index_mode="absolute_datetime",
        )


def test_health_and_response_models_parse_current_payloads() -> None:
    health = HealthMetadata.from_payload(_health_metadata())
    response = ForecastResponse.from_payload(
        {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
            "checkpoint_version": "smoke-1",
            "head": "dummy",
            "query_mode": "forecast",
            "return_mode": "mean",
            "outputs": {
                "query_times": ["2026-01-05T00:00:00+00:00"],
                "requested_columns": ["target"],
                "mean": [[100.0]],
                "samples": None,
                "quantiles": None,
            },
            "diagnostics": {"history_rows": 4, "horizon_count": 1, "seed": 7},
            "errors": [],
        }
    )

    assert health.model_version == "jointfm-inference:0.2.0+ckpt.smoke-1"
    assert isinstance(response, MeanForecastResult)
    assert response.requested_columns == ("target",)
    assert response.mean == ((100.0,),)
    assert response.outputs.requested_columns == ("target",)
    assert response.diagnostics.horizon_count == 1
    assert response.errors == ()


def test_forecast_result_conversion_helpers_cover_mean_samples_and_quantiles() -> None:
    mean_result = ForecastResponse.from_payload(
        {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
            "checkpoint_version": "smoke-1",
            "head": "dummy",
            "query_mode": "forecast",
            "return_mode": "mean",
            "outputs": {
                "query_times": [1, 2],
                "requested_columns": ["target"],
                "mean": [[10.0], [11.0]],
                "samples": None,
                "quantiles": None,
            },
            "diagnostics": {"history_rows": 4, "horizon_count": 2, "seed": 7},
            "errors": [],
        }
    )
    sample_result = ForecastResponse.from_payload(
        {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
            "checkpoint_version": "smoke-1",
            "head": "dummy",
            "query_mode": "forecast",
            "return_mode": "samples",
            "outputs": {
                "query_times": [1, 2],
                "requested_columns": ["target"],
                "mean": None,
                "samples": [
                    [[10.0], [11.0]],
                    [[10.5], [11.5]],
                ],
                "quantiles": None,
            },
            "diagnostics": {"history_rows": 4, "horizon_count": 2, "seed": 7},
            "errors": [],
        }
    )
    quantile_result = ForecastResponse.from_payload(
        {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
            "checkpoint_version": "smoke-1",
            "head": "dummy",
            "query_mode": "forecast",
            "return_mode": "quantiles",
            "outputs": {
                "query_times": [1, 2],
                "requested_columns": ["target"],
                "mean": None,
                "samples": None,
                "quantiles": [
                    {"quantile": 0.1, "values": [[9.0], [10.0]]},
                    {"quantile": 0.9, "values": [[11.0], [12.0]]},
                ],
            },
            "diagnostics": {"history_rows": 4, "horizon_count": 2, "seed": 7},
            "errors": [],
        }
    )

    assert isinstance(mean_result, MeanForecastResult)
    np.testing.assert_allclose(mean_result.to_numpy(), np.array([[10.0], [11.0]]))
    assert mean_result.to_pandas_tidy().to_dict("records") == [
        {"query_time": 1, "requested_column": "target", "value": 10.0},
        {"query_time": 2, "requested_column": "target", "value": 11.0},
    ]
    assert mean_result.to_pandas_wide().to_dict("records") == [
        {"query_time": 1, "target": 10.0},
        {"query_time": 2, "target": 11.0},
    ]

    assert isinstance(sample_result, SampleForecastResult)
    np.testing.assert_allclose(
        sample_result.to_numpy(),
        np.array([[[10.0], [11.0]], [[10.5], [11.5]]]),
    )
    assert sample_result.to_pandas_tidy().to_dict("records") == [
        {"sample": 0, "query_time": 1, "requested_column": "target", "value": 10.0},
        {"sample": 0, "query_time": 2, "requested_column": "target", "value": 11.0},
        {"sample": 1, "query_time": 1, "requested_column": "target", "value": 10.5},
        {"sample": 1, "query_time": 2, "requested_column": "target", "value": 11.5},
    ]
    assert sample_result.to_pandas_wide().to_dict("records") == [
        {"sample": 0, "query_time": 1, "target": 10.0},
        {"sample": 0, "query_time": 2, "target": 11.0},
        {"sample": 1, "query_time": 1, "target": 10.5},
        {"sample": 1, "query_time": 2, "target": 11.5},
    ]

    assert isinstance(quantile_result, QuantileForecastResult)
    assert quantile_result.quantile_levels == (0.1, 0.9)
    np.testing.assert_allclose(
        quantile_result.to_numpy(),
        np.array([[[9.0], [10.0]], [[11.0], [12.0]]]),
    )
    assert quantile_result.to_pandas_tidy().to_dict("records") == [
        {"quantile": 0.1, "query_time": 1, "requested_column": "target", "value": 9.0},
        {"quantile": 0.1, "query_time": 2, "requested_column": "target", "value": 10.0},
        {"quantile": 0.9, "query_time": 1, "requested_column": "target", "value": 11.0},
        {"quantile": 0.9, "query_time": 2, "requested_column": "target", "value": 12.0},
    ]
    assert quantile_result.to_pandas_wide().to_dict("records") == [
        {"quantile": 0.1, "query_time": 1, "target": 9.0},
        {"quantile": 0.1, "query_time": 2, "target": 10.0},
        {"quantile": 0.9, "query_time": 1, "target": 11.0},
        {"quantile": 0.9, "query_time": 2, "target": 12.0},
    ]


def test_forecast_response_validates_request_scoped_shapes() -> None:
    request_payload = {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "query_mode": "forecast",
        "return_mode": "samples",
        "query_times": [1, 2],
        "requested_columns": ["target"],
        "n_samples": 3,
    }
    response_payload = {
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
        "checkpoint_version": "smoke-1",
        "head": "dummy",
        "query_mode": "forecast",
        "return_mode": "samples",
        "outputs": {
            "query_times": [1, 2],
            "requested_columns": ["target"],
            "mean": None,
            "samples": [
                [[10.0], [11.0]],
                [[10.5], [11.5]],
            ],
            "quantiles": None,
        },
        "diagnostics": {"history_rows": 4, "horizon_count": 2, "seed": 7},
        "errors": [],
    }

    with pytest.raises(ValueError, match="outputs.samples length mismatch"):
        ForecastResponse.from_payload(response_payload, request_payload=request_payload)


def test_forecast_response_raises_typed_error_for_success_payload_errors() -> None:
    with pytest.raises(JointFMServiceError) as exc_info:
        ForecastResponse.from_payload(
            {
                "schema_version": "v1",
                "image_version": "0.2.0",
                "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
                "checkpoint_version": "smoke-1",
                "head": "dummy",
                "query_mode": "forecast",
                "return_mode": "mean",
                "errors": [
                    {
                        "code": "VALIDATION_ERROR",
                        "message": "bad request",
                        "field": "query_times",
                    }
                ],
            }
        )

    assert exc_info.value.jointfm_errors == (
        {
            "code": "VALIDATION_ERROR",
            "message": "bad request",
            "field": "query_times",
        },
    )


def test_validate_service_metadata_accepts_current_v1_contract() -> None:
    validate_service_metadata(
        _health_metadata(),
        expected_model_version="jointfm-inference:0.2.0+ckpt.smoke-1",
    )


def test_validate_service_metadata_rejects_schema_mismatch() -> None:
    metadata = _health_metadata()
    metadata["schema_version"] = "v2"

    with pytest.raises(UnsupportedSchemaVersionError, match="schema_version"):
        validate_service_metadata(metadata)


def test_validate_service_metadata_rejects_model_mismatch() -> None:
    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        validate_service_metadata(
            _health_metadata(),
            expected_model_version="jointfm-inference:9.9.9+ckpt.other",
        )


def test_validate_service_metadata_rejects_unknown_advertised_mode() -> None:
    metadata = _health_metadata()
    metadata["supported_return_modes"] = [
        "mean",
        "samples",
        "quantiles",
        "log_prob",
        "median",
    ]

    with pytest.raises(UnsupportedServiceContractError, match="supported_return_modes"):
        validate_service_metadata(metadata)


def test_validate_service_metadata_rejects_malformed_capabilities() -> None:
    metadata = _health_metadata()
    metadata["supported_query_modes"] = "forecast"

    with pytest.raises(UnsupportedServiceContractError, match="supported_query_modes"):
        validate_service_metadata(metadata)

    metadata = _health_metadata()
    metadata["supported_query_modes"] = [""]

    with pytest.raises(UnsupportedServiceContractError, match="supported_query_modes"):
        validate_service_metadata(metadata)


def test_validate_service_metadata_rejects_missing_model_version() -> None:
    metadata = _health_metadata()
    metadata["model_version"] = ""

    with pytest.raises(UnsupportedServiceContractError, match="model_version"):
        validate_service_metadata(metadata)


def test_jointfm_client_methods_fail_fast_without_configuration() -> None:
    client = JointFMClient()

    with pytest.raises(JointFMConfigurationError, match="health"):
        client.health()

    with pytest.raises(JointFMConfigurationError, match="requires settings"):
        client.predict({})

    with pytest.raises(JointFMConfigurationError, match="predict"):
        client.forecast([], query_times=[])


def test_forecast_convenience_methods_share_forecast_path() -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.return_modes: list[str] = []

        def get_json(self, url: str) -> Mapping[str, Any]:
            raise AssertionError(f"unexpected health call to {url}")

        def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
            del url
            return_mode = payload["return_mode"]
            assert isinstance(return_mode, str)
            self.return_modes.append(return_mode)
            return _forecast_response_payload(return_mode=return_mode)

    def _forecast_response_payload(*, return_mode: str) -> dict[str, object]:
        return {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": "jointfm-inference:0.2.0+ckpt.smoke-1",
            "checkpoint_version": "smoke-1",
            "head": "dummy",
            "query_mode": "forecast",
            "return_mode": return_mode,
            "outputs": {
                "query_times": [2],
                "requested_columns": ["target"],
                "mean": [[2.0]] if return_mode == "mean" else None,
                "samples": [[[2.0]]] if return_mode == "samples" else None,
                "quantiles": [
                    {"quantile": 0.1, "values": [[1.0]]},
                    {"quantile": 0.9, "values": [[3.0]]},
                ]
                if return_mode == "quantiles"
                else None,
            },
            "diagnostics": {"history_rows": 1, "horizon_count": 1},
            "errors": [],
        }

    transport = RecordingTransport()
    client = JointFMClient(
        predict_url="http://localhost:8080/predict",
        transport=transport,
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )
    history_rows = [{"target": 1.0}]
    model_version = "jointfm-inference:0.2.0+ckpt.smoke-1"

    assert (
        client.forecast_mean(
            history_rows,
            schema=schema,
            query_times=[2],
            model_version=model_version,
        ).return_mode
        == "mean"
    )
    assert (
        client.forecast_samples(
            history_rows,
            schema=schema,
            query_times=[2],
            model_version=model_version,
        ).return_mode
        == "samples"
    )
    assert (
        client.forecast_quantiles(
            history_rows,
            schema=schema,
            query_times=[2],
            model_version=model_version,
            quantiles=[0.1, 0.9],
        ).return_mode
        == "quantiles"
    )
    assert transport.return_modes == ["mean", "samples", "quantiles"]