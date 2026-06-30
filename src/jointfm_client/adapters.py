# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DataFrame and array adapters for JointFM V1 forecast requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import math
from typing import Any, TypeAlias, cast

from jointfm_client.configuration import DEFAULT_FORECAST_SCHEMA_VERSION
from jointfm_client.contract import (
    DEFAULT_CALENDAR_ID,
    ColumnModality,
    ColumnRole,
    ColumnSpec,
    DataFrameSchema,
    ReturnMode,
    TimeIndexMode,
    TimeValueKind,
    build_forecast_payload,
)
from jointfm_client.contract import (
    _serialize_absolute_datetime,
    _to_json_compatible,
)

ColumnBounds: TypeAlias = Mapping[str, tuple[float | int | None, float | int | None]]
ColumnMapping: TypeAlias = Mapping[str, Mapping[str | int, int]]


def infer_column_specs_from_dataframe(
    frame: Any,
    *,
    time_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    known_dynamic_columns: Sequence[str] | None = None,
    past_dynamic_columns: Sequence[str] | None = None,
    static_columns: Sequence[str] | None = None,
    column_roles: Mapping[str, ColumnRole] | None = None,
    categorical_columns: Sequence[str] | None = None,
    categorical_mappings: ColumnMapping | None = None,
    ordinal_columns: Sequence[str] | None = None,
    ordinal_mappings: ColumnMapping | None = None,
    count_columns: Sequence[str] | None = None,
    time_value_columns: Sequence[str] | Mapping[str, TimeValueKind] | None = None,
    nullable_columns: Sequence[str] | None = None,
    bounds: ColumnBounds | None = None,
) -> tuple[ColumnSpec, ...]:
    """Infer ordered ``ColumnSpec`` objects from a pandas ``DataFrame``."""
    pandas_module = _require_pandas()
    if not isinstance(frame, pandas_module.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if len(frame) == 0:
        raise ValueError("frame must contain at least one row")

    frame_columns = tuple(str(column_name) for column_name in frame.columns)
    if time_column is not None and time_column not in frame_columns:
        raise ValueError(f"time_column {time_column!r} is not present in frame")

    modeled_columns = tuple(
        column_name for column_name in frame_columns if column_name != time_column
    )
    if not modeled_columns:
        raise ValueError("frame must contain at least one non-time column")

    _validate_named_columns(
        modeled_columns,
        {
            "target_columns": target_columns,
            "feature_columns": feature_columns,
            "known_dynamic_columns": known_dynamic_columns,
            "past_dynamic_columns": past_dynamic_columns,
            "static_columns": static_columns,
            "column_roles": None if column_roles is None else tuple(column_roles),
            "categorical_columns": categorical_columns,
            "categorical_mappings": None
            if categorical_mappings is None
            else tuple(categorical_mappings),
            "ordinal_columns": ordinal_columns,
            "ordinal_mappings": None
            if ordinal_mappings is None
            else tuple(ordinal_mappings),
            "count_columns": count_columns,
            "time_value_columns": _time_value_column_names(time_value_columns),
            "nullable_columns": nullable_columns,
            "bounds": None if bounds is None else tuple(bounds),
        },
    )

    role_by_column = _resolve_role_columns(
        modeled_columns,
        target_columns=target_columns,
        feature_columns=feature_columns,
        known_dynamic_columns=known_dynamic_columns,
        past_dynamic_columns=past_dynamic_columns,
        static_columns=static_columns,
        column_roles=column_roles,
    )
    categorical_set = set(categorical_columns or ()) | set(
        categorical_mappings.keys() if categorical_mappings is not None else ()
    )
    ordinal_set = set(ordinal_columns or ()) | set(
        ordinal_mappings.keys() if ordinal_mappings is not None else ()
    )
    count_set = set(count_columns or ())
    nullable_set = set(nullable_columns or ())
    time_value_kinds = _resolve_time_value_kinds(time_value_columns)

    inferred: list[ColumnSpec] = []
    for column_name in modeled_columns:
        series = frame[column_name]
        modality = _infer_modality(
            series,
            column_name=column_name,
            categorical_columns=categorical_set,
            ordinal_columns=ordinal_set,
            count_columns=count_set,
            time_value_columns=set(time_value_kinds),
        )
        mapping = _mapping_for_column(
            series,
            column_name=column_name,
            modality=modality,
            categorical_mappings=categorical_mappings,
            ordinal_mappings=ordinal_mappings,
        )
        lower_bound, upper_bound = _bounds_for_column(column_name, bounds)
        time_value_kind = _time_value_kind_for_column(
            series,
            column_name=column_name,
            modality=modality,
            explicit_kinds=time_value_kinds,
        )
        inferred.append(
            ColumnSpec(
                name=column_name,
                modality=modality,
                role=role_by_column.get(column_name, "feature"),
                nullable=column_name in nullable_set or bool(series.isna().any()),
                vocabulary_size=len(mapping)
                if modality == "categorical" and mapping is not None
                else None,
                level_count=len(mapping)
                if modality == "ordinal" and mapping is not None
                else None,
                mapping=mapping,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                time_value_kind=time_value_kind,
            )
        )
    return tuple(inferred)


def dataframe_to_history_rows(
    frame: Any,
    schema: DataFrameSchema,
) -> list[dict[str, Any]]:
    """Convert a pandas ``DataFrame`` into ordered JointFM ``history_rows``."""
    pandas_module = _require_pandas()
    numpy_module = _require_numpy()
    if not isinstance(frame, pandas_module.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if not isinstance(schema, DataFrameSchema):
        raise ValueError("schema must be a DataFrameSchema")
    if len(frame) == 0:
        raise ValueError("frame must contain at least one row")

    ordered_columns = _ordered_history_columns(schema)
    missing_columns = sorted(set(ordered_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"frame is missing history columns: {missing_columns}")

    column_specs = {column.name: column for column in schema.columns}
    rows: list[dict[str, Any]] = []
    for row_index, row_values in enumerate(
        frame.loc[:, ordered_columns].itertuples(index=False, name=None)
    ):
        row_payload: dict[str, Any] = {}
        for column_name, value in zip(ordered_columns, row_values, strict=True):
            if column_name == schema.time_column:
                row_payload[column_name] = _time_index_value_to_json(
                    value,
                    time_index_mode=schema.time_index_mode,
                    field=f"history_rows[{row_index}].{column_name}",
                    pandas_module=pandas_module,
                    numpy_module=numpy_module,
                )
                continue
            column_spec = column_specs[column_name]
            row_payload[column_name] = _column_value_to_json(
                value,
                column_spec=column_spec,
                field=f"history_rows[{row_index}].{column_name}",
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
        rows.append(row_payload)
    return rows


def arrays_to_history_rows(
    values: Any,
    *,
    columns: Sequence[ColumnSpec],
    time_column: str | None = None,
    time_values: Sequence[Any] | None = None,
    time_index_mode: TimeIndexMode = "ordinal",
) -> list[dict[str, Any]]:
    """Convert a two-dimensional NumPy-like array into ``history_rows``."""
    numpy_module = _require_numpy()
    pandas_module = _require_pandas()
    column_specs = _require_column_specs(columns)
    array = numpy_module.asarray(values, dtype=object)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    row_count, column_count = cast(tuple[int, int], array.shape)
    if row_count == 0:
        raise ValueError("values must contain at least one row")
    if column_count != len(column_specs):
        raise ValueError(
            "values column count must match the number of ColumnSpec objects"
        )
    if (time_column is None) != (time_values is None):
        raise ValueError("time_column and time_values must be provided together")
    if time_values is not None and len(time_values) != row_count:
        raise ValueError("time_values length must match values row count")

    rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        row_payload: dict[str, Any] = {}
        if time_column is not None and time_values is not None:
            row_payload[time_column] = _time_index_value_to_json(
                time_values[row_index],
                time_index_mode=time_index_mode,
                field=f"history_rows[{row_index}].{time_column}",
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
        for column_index, column_spec in enumerate(column_specs):
            row_payload[column_spec.name] = _column_value_to_json(
                array[row_index, column_index],
                column_spec=column_spec,
                field=f"history_rows[{row_index}].{column_spec.name}",
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
        rows.append(row_payload)
    return rows


def build_forecast_payload_from_dataframe(
    frame: Any,
    *,
    model_version: str,
    time_index_mode: TimeIndexMode,
    query_times: Sequence[Any],
    columns: Sequence[ColumnSpec] | None = None,
    time_column: str | None = None,
    requested_columns: Sequence[str | int] | None = None,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = DEFAULT_FORECAST_SCHEMA_VERSION,
    time_scale_seconds: float | int | None = None,
    use_local_normalized_time: bool = False,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone: str | None = None,
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    known_dynamic_columns: Sequence[str] | None = None,
    past_dynamic_columns: Sequence[str] | None = None,
    static_columns: Sequence[str] | None = None,
    column_roles: Mapping[str, ColumnRole] | None = None,
    categorical_columns: Sequence[str] | None = None,
    categorical_mappings: ColumnMapping | None = None,
    ordinal_columns: Sequence[str] | None = None,
    ordinal_mappings: ColumnMapping | None = None,
    count_columns: Sequence[str] | None = None,
    time_value_columns: Sequence[str] | Mapping[str, TimeValueKind] | None = None,
    nullable_columns: Sequence[str] | None = None,
    bounds: ColumnBounds | None = None,
) -> dict[str, Any]:
    """Build a validated forecast payload from a pandas ``DataFrame``."""
    column_specs = (
        infer_column_specs_from_dataframe(
            frame,
            time_column=time_column,
            target_columns=target_columns,
            feature_columns=feature_columns,
            known_dynamic_columns=known_dynamic_columns,
            past_dynamic_columns=past_dynamic_columns,
            static_columns=static_columns,
            column_roles=column_roles,
            categorical_columns=categorical_columns,
            categorical_mappings=categorical_mappings,
            ordinal_columns=ordinal_columns,
            ordinal_mappings=ordinal_mappings,
            count_columns=count_columns,
            time_value_columns=time_value_columns,
            nullable_columns=nullable_columns,
            bounds=bounds,
        )
        if columns is None
        else _require_column_specs(columns)
    )
    schema = DataFrameSchema(
        columns=column_specs,
        time_index_mode=time_index_mode,
        time_column=time_column,
        time_scale_seconds=time_scale_seconds,
        use_local_normalized_time=use_local_normalized_time,
        calendar_id=calendar_id,
        timezone=timezone,
    )
    history_rows = dataframe_to_history_rows(frame, schema)
    normalized_query_times = validate_forecast_horizon(
        _history_times_from_dataframe(frame, schema),
        query_times,
        time_index_mode=time_index_mode,
    )
    return build_forecast_payload(
        model_version=model_version,
        schema=schema,
        history_rows=history_rows,
        query_times=normalized_query_times,
        requested_columns=requested_columns,
        return_mode=return_mode,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
        schema_version=schema_version,
    )


def build_forecast_payload_from_arrays(
    values: Any,
    *,
    model_version: str,
    columns: Sequence[ColumnSpec],
    time_index_mode: TimeIndexMode,
    query_times: Sequence[Any],
    time_column: str | None = None,
    time_values: Sequence[Any] | None = None,
    requested_columns: Sequence[str | int] | None = None,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = DEFAULT_FORECAST_SCHEMA_VERSION,
    time_scale_seconds: float | int | None = None,
    use_local_normalized_time: bool = False,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Build a validated forecast payload from a two-dimensional array."""
    column_specs = _require_column_specs(columns)
    schema = DataFrameSchema(
        columns=column_specs,
        time_index_mode=time_index_mode,
        time_column=time_column,
        time_scale_seconds=time_scale_seconds,
        use_local_normalized_time=use_local_normalized_time,
        calendar_id=calendar_id,
        timezone=timezone,
    )
    history_rows = arrays_to_history_rows(
        values,
        columns=column_specs,
        time_column=time_column,
        time_values=time_values,
        time_index_mode=time_index_mode,
    )
    normalized_query_times = validate_forecast_horizon(
        _history_times_from_array_rows(
            row_count=len(history_rows),
            time_index_mode=time_index_mode,
            time_values=time_values,
        ),
        query_times,
        time_index_mode=time_index_mode,
    )
    return build_forecast_payload(
        model_version=model_version,
        schema=schema,
        history_rows=history_rows,
        query_times=normalized_query_times,
        requested_columns=requested_columns,
        return_mode=return_mode,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
        schema_version=schema_version,
    )


def build_datetime_query_times(
    history_times: Sequence[Any],
    *,
    periods: int,
    frequency: str | timedelta | None = None,
) -> list[str]:
    """Build regular future absolute-datetime query times from history."""
    _require_period_count(periods)
    parsed_history = _parse_ordered_values(
        history_times,
        time_index_mode="absolute_datetime",
        field="history_times",
    )
    if frequency is None:
        if len(parsed_history) < 2:
            raise ValueError("frequency is required when history_times has one value")
        step = parsed_history[-1] - parsed_history[-2]
        if step <= timedelta(0):
            raise ValueError("history_times must be strictly increasing")
        values = [
            parsed_history[-1] + step * offset for offset in range(1, periods + 1)
        ]
        return [
            _serialize_absolute_datetime(value, field=f"query_times[{index}]")
            for index, value in enumerate(values)
        ]

    pandas_module = _require_pandas()
    start = pandas_module.Timestamp(parsed_history[-1])
    offset = (
        pandas_module.to_timedelta(frequency)
        if isinstance(frequency, str)
        else frequency
    )
    values = [start + offset * step_index for step_index in range(1, periods + 1)]
    return [
        _serialize_absolute_datetime(
            value.to_pydatetime(), field=f"query_times[{index}]"
        )
        for index, value in enumerate(values)
    ]


def build_ordinal_query_times(
    history_times: Sequence[Any],
    *,
    periods: int,
    step: int = 1,
) -> list[int]:
    """Build regular future ordinal query times from history."""
    _require_period_count(periods)
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    parsed_history = _parse_ordered_values(
        history_times,
        time_index_mode="ordinal",
        field="history_times",
    )
    last_observed = cast(int, parsed_history[-1])
    return [last_observed + step * offset for offset in range(1, periods + 1)]


def build_continuous_query_times(
    history_times: Sequence[Any],
    *,
    periods: int,
    step: float | int | None = None,
) -> list[float]:
    """Build regular future continuous-float query times from history."""
    _require_period_count(periods)
    parsed_history = _parse_ordered_values(
        history_times,
        time_index_mode="continuous_float",
        field="history_times",
    )
    if step is None:
        if len(parsed_history) < 2:
            raise ValueError("step is required when history_times has one value")
        step_value = float(parsed_history[-1]) - float(parsed_history[-2])
    else:
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise ValueError("step must be numeric")
        step_value = float(step)
    if not math.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be a positive finite number")
    last_observed = float(parsed_history[-1])
    return [last_observed + step_value * offset for offset in range(1, periods + 1)]


def validate_forecast_horizon(
    history_times: Sequence[Any],
    query_times: Sequence[Any],
    *,
    time_index_mode: TimeIndexMode,
) -> list[Any]:
    """Validate that query times are a future, increasing forecast horizon."""
    parsed_history = _parse_ordered_values(
        history_times,
        time_index_mode=time_index_mode,
        field="history_times",
    )
    parsed_queries = _parse_ordered_values(
        query_times,
        time_index_mode=time_index_mode,
        field="query_times",
    )
    if parsed_queries[0] <= parsed_history[-1]:
        raise ValueError("query_times must be strictly after the last observed time")
    if time_index_mode == "absolute_datetime":
        return [
            _serialize_absolute_datetime(value, field=f"query_times[{index}]")
            for index, value in enumerate(parsed_queries)
        ]
    return list(parsed_queries)


def _require_pandas() -> Any:
    """Return pandas or raise with the SDK extra needed by DataFrame helpers."""
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "pandas support requires installing jointfm-client[notebooks]"
        ) from error
    return pandas_module


def _require_numpy() -> Any:
    """Return NumPy from the pandas-backed optional adapter dependency."""
    try:
        import numpy as numpy_module
    except ImportError as error:  # pragma: no cover - exercised only without pandas
        raise RuntimeError(
            "NumPy array support requires installing jointfm-client[notebooks]"
        ) from error
    return numpy_module


def _require_column_specs(columns: Sequence[ColumnSpec]) -> tuple[ColumnSpec, ...]:
    """Validate an ordered sequence of ``ColumnSpec`` objects."""
    if not isinstance(columns, Sequence) or isinstance(
        columns, str | bytes | bytearray
    ):
        raise ValueError("columns must be a sequence of ColumnSpec objects")
    if len(columns) == 0:
        raise ValueError("columns must not be empty")
    column_specs = tuple(columns)
    for column_index, column_spec in enumerate(column_specs):
        if not isinstance(column_spec, ColumnSpec):
            raise ValueError(f"columns[{column_index}] must be a ColumnSpec")
    return column_specs


def _ordered_history_columns(schema: DataFrameSchema) -> list[str]:
    """Return history columns in the same order as the service frame builder."""
    ordered_columns = [column.name for column in schema.columns]
    if schema.time_column is not None:
        return [schema.time_column, *ordered_columns]
    return ordered_columns


def _history_times_from_dataframe(frame: Any, schema: DataFrameSchema) -> list[Any]:
    """Extract ordered history times for local forecast-horizon validation."""
    if schema.time_column is not None:
        return list(frame[schema.time_column].tolist())
    if schema.time_index_mode == "ordinal":
        return list(range(len(frame)))
    raise ValueError(
        f"time_column is required for DataFrame {schema.time_index_mode!r} forecasts"
    )


def _history_times_from_array_rows(
    *,
    row_count: int,
    time_index_mode: TimeIndexMode,
    time_values: Sequence[Any] | None,
) -> list[Any]:
    """Extract array history times for local forecast-horizon validation."""
    if time_values is not None:
        return list(time_values)
    if time_index_mode == "ordinal":
        return list(range(row_count))
    raise ValueError(
        f"time_values are required for array {time_index_mode!r} forecasts"
    )


def _time_index_value_to_json(
    value: Any,
    *,
    time_index_mode: TimeIndexMode,
    field: str,
    pandas_module: Any,
    numpy_module: Any,
) -> Any:
    """Serialize a history time-index value according to schema mode."""
    if _is_missing(value, pandas_module=pandas_module):
        raise ValueError(f"{field} must not be null")
    scalar_value = _to_builtin_scalar(value, numpy_module=numpy_module)
    if time_index_mode == "absolute_datetime":
        return _serialize_absolute_datetime(scalar_value, field=field)
    if time_index_mode == "ordinal":
        if isinstance(scalar_value, bool) or not isinstance(scalar_value, int):
            raise ValueError(f"{field} must be an integer for ordinal mode")
        return scalar_value
    if isinstance(scalar_value, bool) or not isinstance(scalar_value, (int, float)):
        raise ValueError(f"{field} must be numeric for continuous_float mode")
    return _finite_number(scalar_value, field=field)


def _column_value_to_json(
    value: Any,
    *,
    column_spec: ColumnSpec,
    field: str,
    pandas_module: Any,
    numpy_module: Any,
) -> Any:
    """Serialize one modeled DataFrame or array cell into request JSON."""
    if _is_missing(value, pandas_module=pandas_module):
        if not column_spec.nullable:
            raise ValueError(f"{field} is null but column is not nullable")
        return None
    scalar_value = _to_builtin_scalar(value, numpy_module=numpy_module)
    if column_spec.modality in {"categorical", "ordinal"} and column_spec.mapping:
        return _encode_mapped_value(
            scalar_value, mapping=column_spec.mapping, field=field
        )
    if (
        column_spec.modality == "time_value"
        and column_spec.time_value_kind == "absolute_datetime"
    ):
        return _serialize_absolute_datetime(scalar_value, field=field)
    return _to_json_compatible(scalar_value, field=field)


def _encode_mapped_value(
    value: Any,
    *,
    mapping: Mapping[str | int, int],
    field: str,
) -> int:
    """Encode one categorical or ordinal value through a declared mapping."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a string or integer mapped value")
    if value not in mapping:
        raise ValueError(f"{field} value {value!r} is not present in column mapping")
    return mapping[value]


def _to_builtin_scalar(value: Any, *, numpy_module: Any) -> Any:
    """Convert NumPy scalars to Python scalars while preserving datetimes."""
    if isinstance(value, numpy_module.generic):
        return value.item()
    return value


def _is_missing(value: Any, *, pandas_module: Any) -> bool:
    """Return whether a scalar adapter value is pandas-null."""
    return bool(pandas_module.isna(value))


def _finite_number(value: int | float, *, field: str) -> int | float:
    """Validate one finite numeric time value."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_named_columns(
    modeled_columns: Sequence[str],
    named_columns_by_field: Mapping[str, Sequence[str] | None],
) -> None:
    """Reject option columns that are not modeled DataFrame columns."""
    modeled_column_set = set(modeled_columns)
    for field, column_names in named_columns_by_field.items():
        if column_names is None:
            continue
        unknown_columns = sorted(set(column_names).difference(modeled_column_set))
        if unknown_columns:
            raise ValueError(f"{field} references unknown columns: {unknown_columns}")


def _resolve_role_columns(
    modeled_columns: Sequence[str],
    *,
    target_columns: Sequence[str] | None,
    feature_columns: Sequence[str] | None,
    known_dynamic_columns: Sequence[str] | None,
    past_dynamic_columns: Sequence[str] | None,
    static_columns: Sequence[str] | None,
    column_roles: Mapping[str, ColumnRole] | None,
) -> dict[str, ColumnRole]:
    """Resolve explicit role-list and per-column role options."""
    del modeled_columns
    role_by_column: dict[str, ColumnRole] = {}
    role_options: tuple[tuple[ColumnRole, Sequence[str] | None], ...] = (
        ("target", target_columns),
        ("feature", feature_columns),
        ("known_dynamic", known_dynamic_columns),
        ("past_dynamic", past_dynamic_columns),
        ("static", static_columns),
    )
    for role, column_names in role_options:
        for column_name in column_names or ():
            previous_role = role_by_column.get(column_name)
            if previous_role is not None and previous_role != role:
                raise ValueError(
                    f"column {column_name!r} has conflicting roles: "
                    f"{previous_role!r} and {role!r}"
                )
            role_by_column[column_name] = role

    if column_roles is None:
        return role_by_column
    for column_name, role in column_roles.items():
        previous_role = role_by_column.get(column_name)
        if previous_role is not None and previous_role != role:
            raise ValueError(
                f"column {column_name!r} has conflicting roles: "
                f"{previous_role!r} and {role!r}"
            )
        role_by_column[column_name] = role
    return role_by_column


def _infer_modality(
    series: Any,
    *,
    column_name: str,
    categorical_columns: set[str],
    ordinal_columns: set[str],
    count_columns: set[str],
    time_value_columns: set[str],
) -> ColumnModality:
    """Infer one column modality from explicit options and pandas dtype."""
    pandas_module = _require_pandas()
    if column_name in time_value_columns:
        return "time_value"
    if column_name in ordinal_columns:
        return "ordinal"
    if column_name in categorical_columns:
        return "categorical"
    if column_name in count_columns:
        return "count"
    if pandas_module.api.types.is_bool_dtype(series):
        return "binary"
    if pandas_module.api.types.is_datetime64_any_dtype(series):
        return "time_value"
    if pandas_module.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _mapping_for_column(
    series: Any,
    *,
    column_name: str,
    modality: ColumnModality,
    categorical_mappings: ColumnMapping | None,
    ordinal_mappings: ColumnMapping | None,
) -> Mapping[str | int, int] | None:
    """Return explicit or inferred categorical/ordinal mapping metadata."""
    if modality == "categorical":
        if categorical_mappings is not None and column_name in categorical_mappings:
            return dict(categorical_mappings[column_name])
        return _infer_value_mapping(series)
    if modality == "ordinal":
        if ordinal_mappings is None or column_name not in ordinal_mappings:
            return None
        return dict(ordinal_mappings[column_name])
    return None


def _infer_value_mapping(series: Any) -> dict[str | int, int]:
    """Infer stable integer codes for non-null string/integer values."""
    numpy_module = _require_numpy()
    values: list[str | int] = []
    for raw_value in series.dropna().unique().tolist():
        value = _to_builtin_scalar(raw_value, numpy_module=numpy_module)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                "categorical inference supports only string or integer values; "
                "pass explicit ColumnSpec objects for other encodings"
            )
        values.append(value)
    return {
        value: value_index
        for value_index, value in enumerate(
            sorted(set(values), key=lambda item: (type(item).__name__, repr(item)))
        )
    }


def _bounds_for_column(
    column_name: str,
    bounds: ColumnBounds | None,
) -> tuple[float | int | None, float | int | None]:
    """Return lower and upper bounds for one column."""
    if bounds is None or column_name not in bounds:
        return None, None
    lower_bound, upper_bound = bounds[column_name]
    return lower_bound, upper_bound


def _time_value_column_names(
    time_value_columns: Sequence[str] | Mapping[str, TimeValueKind] | None,
) -> Sequence[str] | None:
    """Return time-valued column names from either accepted option shape."""
    if time_value_columns is None:
        return None
    if isinstance(time_value_columns, Mapping):
        return tuple(time_value_columns)
    return time_value_columns


def _resolve_time_value_kinds(
    time_value_columns: Sequence[str] | Mapping[str, TimeValueKind] | None,
) -> dict[str, TimeValueKind | None]:
    """Resolve explicitly requested time-valued columns and optional kinds."""
    if time_value_columns is None:
        return {}
    if isinstance(time_value_columns, Mapping):
        kind_by_column = cast(Mapping[str, TimeValueKind], time_value_columns)
        return {column_name: kind for column_name, kind in kind_by_column.items()}
    return {column_name: None for column_name in time_value_columns}


def _time_value_kind_for_column(
    series: Any,
    *,
    column_name: str,
    modality: ColumnModality,
    explicit_kinds: Mapping[str, TimeValueKind | None],
) -> TimeValueKind | None:
    """Infer or return the configured time-value kind for one column."""
    if modality != "time_value":
        return None
    explicit_kind = explicit_kinds.get(column_name)
    if explicit_kind is not None:
        return explicit_kind
    pandas_module = _require_pandas()
    if pandas_module.api.types.is_datetime64_any_dtype(series):
        return "absolute_datetime"
    return "continuous_float"


def _require_period_count(periods: int) -> None:
    """Validate a positive forecast horizon period count."""
    if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
        raise ValueError("periods must be a positive integer")


def _parse_ordered_values(
    values: Sequence[Any],
    *,
    time_index_mode: TimeIndexMode,
    field: str,
) -> list[Any]:
    """Parse and validate an increasing ordered time sequence."""
    if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{field} must be a sequence")
    if len(values) == 0:
        raise ValueError(f"{field} must not be empty")
    parsed = [
        _parse_time_value(
            value, time_index_mode=time_index_mode, field=f"{field}[{index}]"
        )
        for index, value in enumerate(values)
    ]
    for value_index in range(1, len(parsed)):
        if parsed[value_index] <= parsed[value_index - 1]:
            raise ValueError(f"{field} must be strictly monotonically increasing")
    return parsed


def _parse_time_value(value: Any, *, time_index_mode: TimeIndexMode, field: str) -> Any:
    """Parse one value into the comparison type for a time-index mode."""
    if time_index_mode == "absolute_datetime":
        serialized = _serialize_absolute_datetime(value, field=field)
        return datetime.fromisoformat(serialized)
    if time_index_mode == "ordinal":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer for ordinal mode")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric for continuous_float mode")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed
