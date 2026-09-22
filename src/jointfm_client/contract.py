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

"""JointFM service contract constants, models, and compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import math
from typing import Any, Final, Literal, Self, TypeAlias, TypeVar, cast

from jointfm_client.exceptions import (
    JointFMServiceError,
    UnsupportedModelVersionError,
    UnsupportedSchemaVersionError,
    UnsupportedServiceContractError,
)

DISTRIBUTION_NAME: Final = "jointfm-client"
IMPORT_NAMESPACE: Final = "jointfm_client"
FIRST_SUPPORTED_PYTHON_VERSION: Final = "3.11"
# Read from installed distribution metadata so the version cannot drift from
# pyproject.toml the way a hand-maintained literal here did.
PACKAGE_VERSION: Final = importlib.metadata.version(DISTRIBUTION_NAME)

SCHEMA_VERSION: Final = "v3"
# The service sums and averages its log densities in the model's own precision,
# so its summaries differ from a recomputation in the last few digits.
DERIVED_SCORE_TOLERANCE: Final = 1e-6
DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE: Final = (
    "deployments/{deployment_id}/predictionsUnstructured"
)
LOCAL_HEALTH_ROUTE: Final = "/healthz"
LOCAL_PREDICT_ROUTE: Final = "/predict"
DEFAULT_CALENDAR_ID: Final = "pandas-default"

PREDICT_REQUEST_TYPE: Final = "predict"
HEALTH_REQUEST_TYPE: Final = "health"
SUPPORTED_REQUEST_TYPES: Final[tuple[str, ...]] = (
    PREDICT_REQUEST_TYPE,
    HEALTH_REQUEST_TYPE,
)

QueryMode: TypeAlias = Literal["forecast", "condition"]
ConditionKind: TypeAlias = Literal["equality", "interval"]
ReturnMode: TypeAlias = Literal["mean", "samples", "quantiles", "log_prob"]
TimeIndexMode: TypeAlias = Literal[
    "ordinal",
    "continuous_float",
    "absolute_datetime",
]
DecodingStrategy: TypeAlias = Literal[
    "parallel_dense",
    "parallel_scalable",
    "autoregressive",
]
ColumnModality: TypeAlias = Literal[
    "numeric",
    "categorical",
    "ordinal",
    "binary",
    "count",
    "time_value",
]
ColumnRole: TypeAlias = Literal[
    "target",
    "known_dynamic",
    "past_dynamic",
    "static",
    "feature",
]
TimeValueKind: TypeAlias = Literal["continuous_float", "absolute_datetime"]
StructuredErrorCode: TypeAlias = Literal[
    "VALIDATION_ERROR",
    "UNSUPPORTED_HEAD_QUERY_COMBINATION",
    "UNSUPPORTED_RETURN_MODE",
    "SCHEMA_VERSION_MISMATCH",
    "MODEL_VERSION_MISMATCH",
    "INPUT_SIZE_EXCEEDED",
    "INTERNAL_ERROR",
]

SUPPORTED_QUERY_MODES: Final[tuple[QueryMode, ...]] = ("forecast", "condition")
SUPPORTED_CONDITION_KINDS: Final[tuple[ConditionKind, ...]] = ("equality", "interval")
SUPPORTED_RETURN_MODES: Final[tuple[ReturnMode, ...]] = (
    "mean",
    "samples",
    "quantiles",
    "log_prob",
)
SUPPORTED_TIME_INDEX_MODES: Final[tuple[TimeIndexMode, ...]] = (
    "ordinal",
    "continuous_float",
    "absolute_datetime",
)
SUPPORTED_DECODING_STRATEGIES: Final[tuple[DecodingStrategy, ...]] = (
    "parallel_dense",
    "parallel_scalable",
    "autoregressive",
)
SUPPORTED_COLUMN_MODALITIES: Final[tuple[ColumnModality, ...]] = (
    "numeric",
    "categorical",
    "ordinal",
    "binary",
    "count",
    "time_value",
)
SUPPORTED_COLUMN_ROLES: Final[tuple[ColumnRole, ...]] = (
    "target",
    "known_dynamic",
    "past_dynamic",
    "static",
    "feature",
)
SUPPORTED_TIME_VALUE_KINDS: Final[tuple[TimeValueKind, ...]] = (
    "continuous_float",
    "absolute_datetime",
)
STRUCTURED_ERROR_CODES: Final[tuple[StructuredErrorCode, ...]] = (
    "VALIDATION_ERROR",
    "UNSUPPORTED_HEAD_QUERY_COMBINATION",
    "UNSUPPORTED_RETURN_MODE",
    "SCHEMA_VERSION_MISMATCH",
    "MODEL_VERSION_MISMATCH",
    "INPUT_SIZE_EXCEEDED",
    "INTERNAL_ERROR",
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """JSON-facing descriptor for one JointFM request column."""

    name: str
    modality: ColumnModality
    role: ColumnRole = "feature"
    nullable: bool = False
    vocabulary_size: int | None = None
    level_count: int | None = None
    mapping: Mapping[str | int, int] | None = None
    lower_bound: float | int | None = None
    upper_bound: float | int | None = None
    time_value_kind: TimeValueKind | None = None
    time_value_scale_seconds: float | int | None = None
    time_value_use_local_normalized_time: bool = False
    time_value_calendar_id: str = DEFAULT_CALENDAR_ID
    time_value_timezone: str | None = None

    def __post_init__(self) -> None:
        """Validate column metadata against the service contract."""
        _require_string(self.name, field="columns.name")
        _require_member(
            self.modality,
            field="columns.modality",
            supported_values=SUPPORTED_COLUMN_MODALITIES,
        )
        _require_member(
            self.role,
            field="columns.role",
            supported_values=SUPPORTED_COLUMN_ROLES,
        )
        _require_bool(self.nullable, field="columns.nullable")
        _optional_positive_int(self.vocabulary_size, field="columns.vocabulary_size")
        _optional_positive_int(self.level_count, field="columns.level_count")
        _optional_mapping(self.mapping, field="columns.mapping")
        lower_bound = _optional_float(self.lower_bound, field="columns.lower_bound")
        upper_bound = _optional_float(self.upper_bound, field="columns.upper_bound")
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("columns.lower_bound must not exceed columns.upper_bound")

        _optional_member(
            self.time_value_kind,
            field="columns.time_value_kind",
            supported_values=SUPPORTED_TIME_VALUE_KINDS,
        )
        _optional_positive_float(
            self.time_value_scale_seconds,
            field="columns.time_value_scale_seconds",
        )
        _require_bool(
            self.time_value_use_local_normalized_time,
            field="columns.time_value_use_local_normalized_time",
        )
        _require_string(
            self.time_value_calendar_id,
            field="columns.time_value_calendar_id",
        )
        _optional_string(self.time_value_timezone, field="columns.time_value_timezone")
        _validate_time_value_options(self)

    def to_payload(self) -> dict[str, Any]:
        """Return this column descriptor as a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "name": self.name,
            "modality": self.modality,
        }
        if self.role != "feature":
            payload["role"] = self.role
        if self.nullable:
            payload["nullable"] = self.nullable
        if self.vocabulary_size is not None:
            payload["vocabulary_size"] = self.vocabulary_size
        if self.level_count is not None:
            payload["level_count"] = self.level_count
        if self.mapping is not None:
            payload["mapping"] = dict(self.mapping)
        if self.lower_bound is not None:
            payload["lower_bound"] = _optional_float(
                self.lower_bound,
                field="columns.lower_bound",
            )
        if self.upper_bound is not None:
            payload["upper_bound"] = _optional_float(
                self.upper_bound,
                field="columns.upper_bound",
            )
        if self.time_value_kind is not None:
            payload["time_value_kind"] = self.time_value_kind
        if self.time_value_scale_seconds is not None:
            payload["time_value_scale_seconds"] = _optional_float(
                self.time_value_scale_seconds,
                field="columns.time_value_scale_seconds",
            )
        if self.time_value_use_local_normalized_time:
            payload["time_value_use_local_normalized_time"] = True
        if self.time_value_calendar_id != DEFAULT_CALENDAR_ID:
            payload["time_value_calendar_id"] = self.time_value_calendar_id
        if self.time_value_timezone is not None:
            payload["time_value_timezone"] = self.time_value_timezone
        return payload


@dataclass(frozen=True, slots=True)
class DataFrameSchema:
    """JSON-facing schema for the tabular history in one forecast request."""

    columns: Sequence[ColumnSpec]
    time_index_mode: TimeIndexMode
    time_column: str | None = None
    time_scale_seconds: float | int | None = None
    use_local_normalized_time: bool = False
    calendar_id: str = DEFAULT_CALENDAR_ID
    timezone: str | None = None

    def __post_init__(self) -> None:
        """Validate schema fields that span multiple columns."""
        columns = _require_sequence(self.columns, field="columns")
        for index, column in enumerate(columns):
            if not isinstance(column, ColumnSpec):
                raise ValueError(f"columns[{index}] must be a ColumnSpec")
        _require_member(
            self.time_index_mode,
            field="time_index_mode",
            supported_values=SUPPORTED_TIME_INDEX_MODES,
        )
        _optional_string(self.time_column, field="time_column")
        _optional_positive_float(self.time_scale_seconds, field="time_scale_seconds")
        _require_bool(
            self.use_local_normalized_time,
            field="use_local_normalized_time",
        )
        _require_string(self.calendar_id, field="calendar_id")
        _optional_string(self.timezone, field="timezone")

        column_names = [column.name for column in columns]
        if len(set(column_names)) != len(column_names):
            raise ValueError("columns must not contain duplicate names")
        if self.time_column is not None and self.time_column in set(column_names):
            raise ValueError("time_column must not also appear in columns")
        if self.time_index_mode == "absolute_datetime" and self.time_column is None:
            raise ValueError("absolute_datetime requests must provide time_column")

    def to_payload(self) -> dict[str, Any]:
        """Return this schema as the JSON fields expected by `/predict`."""
        payload: dict[str, Any] = {
            "time_index_mode": self.time_index_mode,
            "columns": [column.to_payload() for column in self.columns],
        }
        if self.time_column is not None:
            payload["time_column"] = self.time_column
        if self.time_scale_seconds is not None:
            payload["time_scale_seconds"] = _optional_float(
                self.time_scale_seconds,
                field="time_scale_seconds",
            )
        if self.use_local_normalized_time:
            payload["use_local_normalized_time"] = True
        if self.calendar_id != DEFAULT_CALENDAR_ID:
            payload["calendar_id"] = self.calendar_id
        if self.timezone is not None:
            payload["timezone"] = self.timezone
        return payload


@dataclass(frozen=True, slots=True)
class EqualityCondition:
    """One column of the conditioned position pinned to a value.

    An equality condition is an event of probability zero under a continuous
    head, so the deployment answers it analytically rather than by filtering
    draws. It fixes the column's value, which is why that column leaves the
    read-out projection.
    """

    column: str
    value: float

    def __post_init__(self) -> None:
        """Reject a pin the service would refuse anyway."""
        _require_string(self.column, field="condition.column")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("condition value must be a number")
        if not math.isfinite(float(self.value)):
            raise ValueError("condition value must be finite")

    def to_payload(self) -> dict[str, Any]:
        """Return this condition as JSON-compatible payload fields."""
        return {"column": self.column, "kind": "equality", "value": float(self.value)}


@dataclass(frozen=True, slots=True)
class IntervalCondition:
    """One column of the conditioned position confined to a range.

    ``None`` on either side leaves it open. The region carries positive mass,
    so the deployment answers it by composition on the reduced mixture, and the
    column stays readable: what it returns is its distribution inside the
    region.
    """

    column: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        """Reject a range that bounds nothing or bounds it backwards."""
        _require_string(self.column, field="condition.column")
        for bound, name in ((self.lower, "lower"), (self.upper, "upper")):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"condition {name} bound must be a number or None")
            if not math.isfinite(float(bound)):
                raise ValueError(f"condition {name} bound must be finite or None")
        if self.lower is None and self.upper is None:
            raise ValueError(
                "an interval open on both sides conditions nothing; give at least one bound"
            )
        if (
            self.lower is not None
            and self.upper is not None
            and float(self.lower) >= float(self.upper)
        ):
            raise ValueError(
                f"condition needs lower < upper, got [{self.lower}, {self.upper}]"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return this condition as JSON-compatible payload fields."""
        return {
            "column": self.column,
            "kind": "interval",
            "lower": None if self.lower is None else float(self.lower),
            "upper": None if self.upper is None else float(self.upper),
        }


@dataclass(frozen=True, slots=True)
class ConditionBlock:
    """Every condition of one request, against one future position.

    The position is named once, by its index into the request's
    ``query_times``, because conditioning relates variables to each other
    within a position and never across horizons. A statement spanning
    horizons is out of reach of this mode and cannot be expressed here.
    """

    query_time_index: int
    conditions: Sequence[EqualityCondition | IntervalCondition]

    def __post_init__(self) -> None:
        """Validate the block independently of any schema or deployment."""
        if isinstance(self.query_time_index, bool) or not isinstance(
            self.query_time_index, int
        ):
            raise ValueError("query_time_index must be an integer")
        if self.query_time_index < 0:
            raise ValueError("query_time_index must not be negative")
        conditions = _require_sequence(self.conditions, field="conditions")
        if not conditions:
            raise ValueError("a condition block must carry at least one condition")
        seen: set[str] = set()
        for index, condition in enumerate(conditions):
            if not isinstance(condition, (EqualityCondition, IntervalCondition)):
                raise ValueError(
                    f"conditions[{index}] must be an EqualityCondition or an "
                    "IntervalCondition"
                )
            if condition.column in seen:
                raise ValueError(
                    f"column {condition.column!r} carries more than one condition"
                )
            seen.add(condition.column)

    @property
    def pinned_columns(self) -> tuple[str, ...]:
        """Columns an equality condition fixes, which leave the read-out set."""
        return tuple(
            condition.column
            for condition in self.conditions
            if isinstance(condition, EqualityCondition)
        )

    @property
    def conditioned_columns(self) -> tuple[str, ...]:
        """Every column this block conditions, of either kind."""
        return tuple(condition.column for condition in self.conditions)

    @property
    def kinds(self) -> tuple[ConditionKind, ...]:
        """The condition kinds this block uses, which a deployment must advertise."""
        kinds: list[ConditionKind] = []
        for condition in self.conditions:
            kind: ConditionKind = (
                "equality" if isinstance(condition, EqualityCondition) else "interval"
            )
            if kind not in kinds:
                kinds.append(kind)
        return tuple(kinds)

    def to_payload(self) -> dict[str, Any]:
        """Return the block as JSON-compatible payload fields."""
        return {
            "query_time_index": self.query_time_index,
            "conditions": [condition.to_payload() for condition in self.conditions],
        }


@dataclass(frozen=True, slots=True)
class ForecastRequestMetadata:
    """Version and mode metadata for one forecast request."""

    model_version: str
    schema_version: str = SCHEMA_VERSION
    query_mode: QueryMode = "forecast"
    return_mode: ReturnMode = "mean"

    def __post_init__(self) -> None:
        """Validate the request metadata against the supported SDK contract."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        _require_string(self.model_version, field="model_version")
        _require_member(
            self.query_mode,
            field="query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        )
        _require_member(
            self.return_mode,
            field="return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return request metadata as JSON-compatible payload fields."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "query_mode": self.query_mode,
            "return_mode": self.return_mode,
        }


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """Validated forecast request that can build a `/predict` payload."""

    metadata: ForecastRequestMetadata
    schema: DataFrameSchema
    history_rows: Sequence[Mapping[str, Any]]
    query_times: Sequence[Any]
    requested_columns: Sequence[str | int] | None = None
    n_samples: int | None = None
    quantiles: Sequence[float | int] | None = None
    seed: int | None = None
    query_row_ids: Sequence[int] | None = None
    condition: ConditionBlock | None = None
    query_rows: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validate payload controls and JSON-facing request arrays."""
        if not isinstance(self.metadata, ForecastRequestMetadata):
            raise ValueError("metadata must be a ForecastRequestMetadata")
        if not isinstance(self.schema, DataFrameSchema):
            raise ValueError("schema must be a DataFrameSchema")
        if self.query_row_ids is not None:
            raise ValueError("query_row_ids is not supported for forecast requests")

        history_rows = _require_sequence(self.history_rows, field="history_rows")
        for index, history_row in enumerate(history_rows):
            _require_mapping(history_row, field=f"history_rows[{index}]")
        _validate_history_declared_columns(history_rows, self.schema)
        _serialize_query_times(
            self.query_times, time_index_mode=self.schema.time_index_mode
        )
        _resolve_requested_columns(self.schema.columns, self.requested_columns)
        _optional_positive_int(self.n_samples, field="n_samples")
        _optional_int(self.seed, field="seed")

        if (self.metadata.query_mode == "condition") != (self.condition is not None):
            raise ValueError(
                "query_mode='condition' and a condition block go together: one "
                "without the other means a different request than the caller wrote"
            )
        if self.condition is not None:
            self._validate_condition()

        if self.metadata.return_mode == "quantiles":
            _require_quantiles(self.quantiles)
        elif self.quantiles is not None:
            raise ValueError(
                "quantiles may be provided only when return_mode='quantiles'"
            )

        if (self.metadata.return_mode == "log_prob") != (self.query_rows is not None):
            raise ValueError(
                "return_mode='log_prob' and query_rows go together: scoring needs "
                "the observed values, and no other return mode reads them"
            )
        if self.query_rows is not None:
            self._validate_query_rows()

    def _validate_condition(self) -> None:
        """Check the condition block against this request's schema and horizons.

        The deployment rejects all of this too, before executing anything; the
        point of repeating it here is that a caller learns which column name it
        got wrong without paying for a round trip.
        """
        block = self.condition
        assert block is not None
        declared = {column.name for column in self.schema.columns}
        unknown = [name for name in block.conditioned_columns if name not in declared]
        if unknown:
            raise ValueError(f"condition references undeclared columns {unknown}")
        horizon_count = len(_require_sequence(self.query_times, field="query_times"))
        if block.query_time_index >= horizon_count:
            raise ValueError(
                f"condition.query_time_index {block.query_time_index} is outside the "
                f"{horizon_count} requested future positions"
            )
        if len(set(block.conditioned_columns)) >= len(declared):
            if len(block.pinned_columns) == len(set(block.conditioned_columns)):
                raise ValueError(
                    "pinning every column leaves nothing to read out; the joint "
                    "density of those values is what return_mode='log_prob' answers"
                )
            raise ValueError(
                "a request must leave at least one column unconditioned: what it "
                "reads out is the conditional distribution of the columns it does "
                "not condition"
            )
        requested = _resolve_requested_columns(
            self.schema.columns, self.requested_columns
        )
        if requested is None:
            return
        requested_names = {value for value in requested if isinstance(value, str)}
        pinned = sorted(set(block.pinned_columns) & requested_names)
        if pinned:
            raise ValueError(
                f"requested_columns lists pinned columns {pinned}; an equality "
                "condition fixes the value, so reading it back returns only what "
                "the request supplied"
            )

    def _validate_query_rows(self) -> None:
        """Check the observed rows a log-density request scores.

        The service scores the whole joint at each future position, so a row
        must carry every declared column and ``requested_columns`` must name
        every readable one in schema order: a narrower projection would score a
        different distribution than the caller believes it asked about. An
        equality condition is the one exception, since it fixes its column's
        value and the conditioning contract forbids reading it back.

        Only two per-value failures are decidable without the deployment's own
        encoding and are checked here — a missing value and a non-finite number.
        A categorical label passes, because the service maps it through the
        column's declared mapping before scoring it.
        """
        rows = _require_sequence(self.query_rows, field="query_rows")
        query_times = _require_sequence(self.query_times, field="query_times")
        if len(rows) != len(query_times):
            raise ValueError(
                f"query_rows must carry one row per query time: got {len(rows)} "
                f"rows for {len(query_times)} query_times"
            )
        declared = [column.name for column in self.schema.columns]
        required = (
            declared
            if self.schema.time_column is None
            else [
                *declared,
                self.schema.time_column,
            ]
        )
        for index, row in enumerate(rows):
            row_mapping = _require_mapping(row, field=f"query_rows[{index}]")
            missing = [name for name in required if name not in row_mapping]
            if missing:
                raise ValueError(
                    f"query_rows[{index}] is missing declared columns: {missing}"
                )
            for name in declared:
                _require_scored_value(
                    row_mapping[name], field=f"query_rows[{index}].{name}"
                )

        requested = _resolve_requested_columns(
            self.schema.columns, self.requested_columns
        )
        if requested is None:
            return
        pinned = set(() if self.condition is None else self.condition.pinned_columns)
        readable = [name for name in declared if name not in pinned]
        if requested != readable:
            excused = (
                f", the pinned columns {sorted(pinned)} excepted" if pinned else ""
            )
            raise ValueError(
                "return_mode='log_prob' scores the whole joint, so "
                "requested_columns must list every declared column in schema "
                f"order{excused}: expected {readable}, got {requested}"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible forecast request without mutating inputs."""
        payload = self.metadata.to_payload()
        payload.update(self.schema.to_payload())
        payload["history_rows"] = _serialize_rows(
            self.history_rows, field="history_rows"
        )
        payload["query_times"] = _serialize_query_times(
            self.query_times,
            time_index_mode=self.schema.time_index_mode,
        )
        requested_columns = _resolve_requested_columns(
            self.schema.columns,
            self.requested_columns,
        )
        if requested_columns is not None:
            payload["requested_columns"] = requested_columns
        if self.n_samples is not None:
            payload["n_samples"] = self.n_samples
        if self.quantiles is not None:
            payload["quantiles"] = _require_quantiles(self.quantiles)
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.condition is not None:
            payload["condition"] = self.condition.to_payload()
        if self.query_rows is not None:
            payload["query_rows"] = _serialize_rows(self.query_rows, field="query_rows")
        return payload


@dataclass(frozen=True, slots=True)
class DataGenerationCapabilities:
    """Training-time data-generation envelope advertised by the deployment.

    Mirrors the inference service's ``data_generation`` health block.
    ``min_series`` and ``max_series`` bound the *total* width of one request —
    feature and target columns counted together — because training draws that
    total and then splits it randomly, guaranteeing only that at least one
    column is a target. There is therefore no separate per-role cap: a request
    is admissible on width alone. ``n_input`` is the longest history and
    ``n_output`` the longest forecast horizon the deployment was trained to
    handle; both are upper bounds, and smaller requests are accepted.
    """

    sampler_type: str
    min_series: int
    max_series: int
    t_input: float
    t_output: float
    n_input: int
    n_output: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse one ``data_generation`` block from a `/healthz` payload.

        Raises ``ValueError`` when a field is missing or malformed, or when the
        series bounds are inverted — an envelope no request could satisfy.
        """
        min_series = _require_positive_int(
            payload.get("min_series"),
            field="data_generation.min_series",
        )
        max_series = _require_positive_int(
            payload.get("max_series"),
            field="data_generation.max_series",
        )
        if min_series > max_series:
            raise ValueError(
                "data_generation.min_series must not exceed "
                f"data_generation.max_series; got {min_series} > {max_series}"
            )
        return cls(
            sampler_type=_require_string(
                payload.get("sampler_type"),
                field="data_generation.sampler_type",
            ),
            min_series=min_series,
            max_series=max_series,
            t_input=_require_positive_float(
                payload.get("t_input"),
                field="data_generation.t_input",
            ),
            t_output=_require_positive_float(
                payload.get("t_output"),
                field="data_generation.t_output",
            ),
            n_input=_require_positive_int(
                payload.get("n_input"),
                field="data_generation.n_input",
            ),
            n_output=_require_positive_int(
                payload.get("n_output"),
                field="data_generation.n_output",
            ),
        )


@dataclass(frozen=True, slots=True)
class HealthMetadata:
    """Typed representation of `/healthz` service metadata."""

    status: str
    schema_version: str
    image_version: str
    model_version: str
    checkpoint_version: str
    checkpoint_path: str
    device: str
    head: str
    decoding_strategy: DecodingStrategy
    supported_query_modes: tuple[str, ...]
    supported_condition_kinds: tuple[str, ...]
    supported_return_modes: tuple[str, ...]
    supported_time_index_modes: tuple[str, ...]
    time_index_encoding: str
    max_sample_count: int
    data_generation: DataGenerationCapabilities | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse and validate a JSON health payload from the service."""
        validate_service_metadata(payload)
        raw_data_generation = payload.get("data_generation")
        if raw_data_generation is None:
            parsed_data_generation: DataGenerationCapabilities | None = None
        else:
            parsed_data_generation = DataGenerationCapabilities.from_payload(
                _require_mapping(raw_data_generation, field="data_generation")
            )
        return cls(
            status=_require_string(payload.get("status"), field="status"),
            schema_version=_require_string(
                payload.get("schema_version"),
                field="schema_version",
            ),
            image_version=_require_string(
                payload.get("image_version"),
                field="image_version",
            ),
            model_version=_require_string(
                payload.get("model_version"),
                field="model_version",
            ),
            checkpoint_version=_require_string(
                payload.get("checkpoint_version"),
                field="checkpoint_version",
            ),
            checkpoint_path=_require_string(
                payload.get("checkpoint_path"),
                field="checkpoint_path",
            ),
            device=_require_string(payload.get("device"), field="device"),
            head=_require_string(payload.get("head"), field="head"),
            decoding_strategy=_require_decoding_strategy(
                payload.get("decoding_strategy")
            ),
            supported_query_modes=_string_tuple(
                payload.get("supported_query_modes"),
                field="supported_query_modes",
            ),
            # A deployment that cannot condition advertises an empty list,
            # which is an answer rather than a missing field.
            supported_condition_kinds=_string_tuple(
                payload.get("supported_condition_kinds"),
                field="supported_condition_kinds",
                allow_empty=True,
            ),
            supported_return_modes=_string_tuple(
                payload.get("supported_return_modes"),
                field="supported_return_modes",
            ),
            supported_time_index_modes=_string_tuple(
                payload.get("supported_time_index_modes"),
                field="supported_time_index_modes",
            ),
            time_index_encoding=_require_string(
                payload.get("time_index_encoding"),
                field="time_index_encoding",
            ),
            max_sample_count=_require_positive_int(
                payload.get("max_sample_count"),
                field="max_sample_count",
            ),
            data_generation=parsed_data_generation,
        )


@dataclass(frozen=True, slots=True)
class StructuredError:
    """Structured error entry returned by the JointFM service."""

    code: str
    message: str
    field: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse one structured service error payload."""
        field = payload.get("field")
        return cls(
            code=_require_string(payload.get("code"), field="errors.code"),
            message=_require_string(payload.get("message"), field="errors.message"),
            field=_optional_string(field, field="errors.field"),
        )


@dataclass(frozen=True, slots=True)
class IntervalEstimator:
    """How a multi-column region's probability was estimated.

    Present only when the interval block spans more than one column, which is
    the only case where a component's box probability has to be estimated; one
    interval column is exact, and a number that never varies would say nothing.
    """

    points: int
    effective_sample_size: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse the estimator accounting from a response payload."""
        return cls(
            points=_require_positive_int(
                payload.get("points"),
                field="diagnostics.interval_estimator.points",
            ),
            effective_sample_size=_require_output_float(
                payload.get("effective_sample_size"),
                field="diagnostics.interval_estimator.effective_sample_size",
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionPlausibility:
    """What the model thinks of the request it was asked to condition on.

    Without it a caller cannot separate "the model is confident" from "you
    conditioned on something the model finds absurd". Both numbers are in log
    space, and each is ``None`` when the request carried no condition of that
    kind. The service reports them and never refuses on them, so acting on them
    is the caller's decision.

    Each number is joint over its whole condition block rather than one value
    per column: ``equality_log_density`` covers every pinned value at once, and
    ``region_log_probability`` covers the whole box at once. Do not rebuild
    either by multiplying per-column numbers, which would assume the columns are
    independent and so discard what a joint model is for.

    ``region_log_probability`` is measured *after* the pinned columns are
    applied, so on a request carrying both kinds it is the log probability of
    the box **given the pins**, not the box's own. The two therefore chain, and
    their sum is the plausibility of the whole condition set::

        equality_log_density + region_log_probability
            == log( density(pins) * P(box | pins) )

    That sum is a density times a probability. Its ``exp`` is not a
    probability -- it carries the reciprocal units of the pinned columns and can
    exceed one -- so compare it, never threshold it: the difference between two
    scenarios pinning the same columns is a log likelihood ratio and is
    unit-free. For a request carrying only interval conditions there is nothing
    to combine and ``region_log_probability`` alone is the joint probability of
    the conditioned event.
    """

    equality_log_density: float | None = None
    region_log_probability: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse the plausibility block from a response payload."""
        equality = payload.get("equality_log_density")
        region = payload.get("region_log_probability")
        return cls(
            equality_log_density=(
                None
                if equality is None
                else _require_output_float(
                    equality, field="plausibility.equality_log_density"
                )
            ),
            region_log_probability=(
                None
                if region is None
                else _require_output_float(
                    region, field="plausibility.region_log_probability"
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ForecastDiagnostics:
    """Diagnostics block returned with one forecast response."""

    history_rows: int
    horizon_count: int
    seed: int | None = None
    condition_draws: int | None = None
    interval_estimator: IntervalEstimator | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Parse response diagnostics from a service payload."""
        raw_estimator = payload.get("interval_estimator")
        return cls(
            condition_draws=_optional_positive_int(
                payload.get("condition_draws"),
                field="diagnostics.condition_draws",
            ),
            interval_estimator=(
                None
                if raw_estimator is None
                else IntervalEstimator.from_payload(
                    _require_mapping(
                        raw_estimator, field="diagnostics.interval_estimator"
                    )
                )
            ),
            history_rows=_require_positive_int(
                payload.get("history_rows"),
                field="diagnostics.history_rows",
            ),
            horizon_count=_require_positive_int(
                payload.get("horizon_count"),
                field="diagnostics.horizon_count",
            ),
            seed=_optional_int(payload.get("seed"), field="diagnostics.seed"),
        )


@dataclass(frozen=True, slots=True)
class QuantileForecast:
    """One quantile surface aligned to the response horizon and requested columns."""

    quantile: float
    values: tuple[tuple[float, ...], ...]

    def to_numpy(self) -> Any:
        """Return NumPy values with axis order ``(horizon, column)``."""
        numpy_module = _require_numpy_module()
        return numpy_module.asarray(self.values, dtype=float)


@dataclass(frozen=True, slots=True)
class LogProbScores:
    """Per-horizon log densities of one scored request, with its summaries.

    The service reports the same numbers several times over: ``values`` holds
    the log density of each scored future position, ``nll_values`` their
    negatives, and the four scalars the sum and the mean of each. Parsing
    recomputes every redundant field from ``values`` instead of trusting it, so
    a truncated or mismatched payload fails here rather than reading as a score.

    A log density is joint over the columns that were scored, so the block has
    no column axis however wide the request was: one number per horizon.
    """

    values: tuple[float, ...]
    nll_values: tuple[float, ...]
    total: float
    mean: float
    nll_total: float
    nll_mean: float

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_horizon_count: int | None = None,
    ) -> Self:
        """Parse one ``outputs.log_prob`` block and check its derived fields."""
        values = _require_float_vector(
            payload.get("values"),
            field="outputs.log_prob.values",
            expected_length=expected_horizon_count,
        )
        nll_values = _require_float_vector(
            payload.get("nll_values"),
            field="outputs.log_prob.nll_values",
            expected_length=len(values),
        )
        for index, (score, negated) in enumerate(zip(values, nll_values, strict=True)):
            _require_derived_score(
                negated,
                expected=-score,
                field=f"outputs.log_prob.nll_values[{index}]",
            )
        summed = math.fsum(values)
        total = _require_derived_score(
            _require_output_float(payload.get("total"), field="outputs.log_prob.total"),
            expected=summed,
            field="outputs.log_prob.total",
        )
        mean = _require_derived_score(
            _require_output_float(payload.get("mean"), field="outputs.log_prob.mean"),
            expected=summed / len(values),
            field="outputs.log_prob.mean",
        )
        nll_total = _require_derived_score(
            _require_output_float(
                payload.get("nll_total"), field="outputs.log_prob.nll_total"
            ),
            expected=-total,
            field="outputs.log_prob.nll_total",
        )
        nll_mean = _require_derived_score(
            _require_output_float(
                payload.get("nll_mean"), field="outputs.log_prob.nll_mean"
            ),
            expected=-mean,
            field="outputs.log_prob.nll_mean",
        )
        return cls(
            values=values,
            nll_values=nll_values,
            total=total,
            mean=mean,
            nll_total=nll_total,
            nll_mean=nll_mean,
        )


@dataclass(frozen=True, slots=True)
class ForecastOutputs:
    """Legacy-shaped forecast output arrays for one parsed forecast result."""

    query_times: tuple[Any, ...]
    requested_columns: tuple[str, ...]
    mean: tuple[tuple[float, ...], ...] | None = None
    samples: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    quantiles: tuple[QuantileForecast, ...] | None = None
    log_prob: LogProbScores | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        return_mode: ReturnMode,
        expected_horizon_count: int | None = None,
        expected_requested_column_count: int | None = None,
        expected_sample_count: int | None = None,
        expected_quantiles: Sequence[float] | None = None,
    ) -> Self:
        """Parse and validate the `outputs` block from one forecast response."""
        query_times = tuple(
            _require_sequence(payload.get("query_times"), field="outputs.query_times")
        )
        requested_columns = _string_tuple(
            payload.get("requested_columns"),
            field="outputs.requested_columns",
        )
        _validate_length(
            actual_length=len(query_times),
            expected_length=expected_horizon_count,
            field="outputs.query_times",
        )
        _validate_length(
            actual_length=len(requested_columns),
            expected_length=expected_requested_column_count,
            field="outputs.requested_columns",
        )

        horizon_count = len(query_times)
        column_count = len(requested_columns)
        mean: tuple[tuple[float, ...], ...] | None = None
        samples: tuple[tuple[tuple[float, ...], ...], ...] | None = None
        quantiles: tuple[QuantileForecast, ...] | None = None
        log_prob: LogProbScores | None = None

        if return_mode == "mean":
            mean = _require_float_matrix(
                payload.get("mean"),
                field="outputs.mean",
                expected_outer_length=horizon_count,
                expected_inner_length=column_count,
            )
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "samples":
            samples = _require_float_tensor3(
                payload.get("samples"),
                field="outputs.samples",
                expected_outer_length=expected_sample_count,
                expected_middle_length=horizon_count,
                expected_inner_length=column_count,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "quantiles":
            quantiles = _require_quantile_forecasts(
                payload.get("quantiles"),
                field="outputs.quantiles",
                expected_quantiles=expected_quantiles,
                expected_horizon_count=horizon_count,
                expected_column_count=column_count,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("log_prob"), field="outputs.log_prob")
        elif return_mode == "log_prob":
            log_prob = LogProbScores.from_payload(
                _require_mapping(payload.get("log_prob"), field="outputs.log_prob"),
                expected_horizon_count=horizon_count,
            )
            _require_none(payload.get("mean"), field="outputs.mean")
            _require_none(payload.get("samples"), field="outputs.samples")
            _require_none(payload.get("quantiles"), field="outputs.quantiles")
        else:
            # A mode nobody parses must fail here rather than fall through to
            # another mode's branch and be read as that mode's shape.
            raise ValueError(
                f"outputs cannot be parsed for return_mode {return_mode!r}"
            )

        return cls(
            query_times=query_times,
            requested_columns=requested_columns,
            mean=mean,
            samples=samples,
            quantiles=quantiles,
            log_prob=log_prob,
        )


@dataclass(frozen=True, slots=True)
class ForecastResponse:
    """Shared metadata preserved on every parsed forecast result."""

    schema_version: str
    image_version: str
    model_version: str
    checkpoint_version: str
    head: str
    query_mode: QueryMode
    return_mode: ReturnMode
    query_times: tuple[Any, ...]
    requested_columns: tuple[str, ...]
    diagnostics: ForecastDiagnostics
    errors: tuple[StructuredError, ...]
    plausibility: ConditionPlausibility | None

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy outputs view for compatibility with nested accessors."""
        raise NotImplementedError

    def to_numpy(self) -> Any:
        """Return a NumPy view of the parsed forecast values."""
        raise NotImplementedError

    def to_pandas_tidy(self) -> Any:
        """Return a tidy pandas DataFrame for the parsed forecast values."""
        raise NotImplementedError

    def to_pandas_wide(self) -> Any:
        """Return a wide pandas DataFrame for the parsed forecast values."""
        raise NotImplementedError

    @staticmethod
    def raise_for_errors(payload: Mapping[str, Any]) -> None:
        """Raise a typed SDK exception when one response carries JointFM errors."""
        _raise_for_response_errors(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        request_payload: Mapping[str, Any] | None = None,
    ) -> "ForecastResponse":
        """Parse one forecast response into the concrete result class for its mode."""
        _raise_for_response_errors(payload)

        expectations = _forecast_response_expectations(request_payload)
        schema_version = _require_string(
            payload.get("schema_version"), field="schema_version"
        )
        if schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                "Unsupported JointFM schema_version: "
                f"expected {SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        if (
            expectations.schema_version is not None
            and schema_version != expectations.schema_version
        ):
            raise ValueError(
                "forecast response schema_version mismatch: "
                f"expected {expectations.schema_version!r}, got {schema_version!r}"
            )

        model_version = _require_string(
            payload.get("model_version"), field="model_version"
        )
        if (
            expectations.model_version is not None
            and model_version != expectations.model_version
        ):
            raise UnsupportedModelVersionError(
                "Unsupported JointFM model_version: "
                f"expected {expectations.model_version!r}, got {model_version!r}"
            )

        query_mode = _require_member(
            payload.get("query_mode"),
            field="query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        )
        if (
            expectations.query_mode is not None
            and query_mode != expectations.query_mode
        ):
            raise ValueError(
                "forecast response query_mode mismatch: "
                f"expected {expectations.query_mode!r}, got {query_mode!r}"
            )

        return_mode = _require_member(
            payload.get("return_mode"),
            field="return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        )
        if (
            expectations.return_mode is not None
            and return_mode != expectations.return_mode
        ):
            raise ValueError(
                "forecast response return_mode mismatch: "
                f"expected {expectations.return_mode!r}, got {return_mode!r}"
            )

        outputs_payload = _require_mapping(payload.get("outputs"), field="outputs")
        outputs = ForecastOutputs.from_payload(
            outputs_payload,
            return_mode=return_mode,
            expected_horizon_count=expectations.horizon_count,
            expected_requested_column_count=expectations.requested_column_count,
            expected_sample_count=expectations.n_samples,
            expected_quantiles=expectations.quantiles,
        )
        diagnostics = ForecastDiagnostics.from_payload(
            _require_mapping(payload.get("diagnostics"), field="diagnostics")
        )
        if diagnostics.horizon_count != len(outputs.query_times):
            raise ValueError(
                "diagnostics.horizon_count mismatch: "
                f"expected {len(outputs.query_times)}, got {diagnostics.horizon_count}"
            )
        if (
            expectations.query_times is not None
            and outputs.query_times != expectations.query_times
        ):
            raise ValueError(
                "forecast response query_times mismatch: "
                f"expected {list(expectations.query_times)!r}, got {list(outputs.query_times)!r}"
            )
        if (
            expectations.requested_columns is not None
            and outputs.requested_columns != expectations.requested_columns
        ):
            raise ValueError(
                "forecast response requested_columns mismatch: "
                f"expected {list(expectations.requested_columns)!r}, "
                f"got {list(outputs.requested_columns)!r}"
            )
        if return_mode == "samples" and outputs.samples is not None:
            _validate_sample_bounds(
                outputs.samples,
                outputs.requested_columns,
                request_payload,
            )

        shared_fields = {
            "schema_version": schema_version,
            "image_version": _require_string(
                payload.get("image_version"), field="image_version"
            ),
            "model_version": model_version,
            "checkpoint_version": _require_string(
                payload.get("checkpoint_version"),
                field="checkpoint_version",
            ),
            "head": _require_string(payload.get("head"), field="head"),
            "query_mode": query_mode,
            "return_mode": return_mode,
            "query_times": outputs.query_times,
            "requested_columns": outputs.requested_columns,
            "diagnostics": diagnostics,
            "errors": _structured_error_tuple(payload.get("errors")),
            "plausibility": (
                None
                if payload.get("plausibility") is None
                else ConditionPlausibility.from_payload(
                    _require_mapping(payload.get("plausibility"), field="plausibility")
                )
            ),
        }

        if return_mode == "mean":
            assert outputs.mean is not None
            return MeanForecastResult(mean=outputs.mean, **shared_fields)
        if return_mode == "samples":
            assert outputs.samples is not None
            return SampleForecastResult(samples=outputs.samples, **shared_fields)
        if return_mode == "quantiles":
            assert outputs.quantiles is not None
            return QuantileForecastResult(quantiles=outputs.quantiles, **shared_fields)
        assert outputs.log_prob is not None
        return LogProbResult(log_prob=outputs.log_prob, **shared_fields)


@dataclass(frozen=True, slots=True)
class MeanForecastResult(ForecastResponse):
    """Parsed mean forecast values with shared metadata and conversion helpers."""

    mean: tuple[tuple[float, ...], ...]

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for mean forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            mean=self.mean,
        )

    def to_numpy(self) -> Any:
        """Return NumPy values with axis order ``(horizon, column)``."""
        numpy_module = _require_numpy_module()
        return numpy_module.asarray(self.mean, dtype=float)

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``query_time``, ``requested_column``, and ``value``."""
        pandas_module = _require_pandas_module()
        rows = [
            {
                "query_time": query_time,
                "requested_column": column_name,
                "value": value,
            }
            for query_time, values_by_column in zip(
                self.query_times, self.mean, strict=True
            )
            for column_name, value in zip(
                self.requested_columns,
                values_by_column,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per forecast horizon."""
        pandas_module = _require_pandas_module()
        rows = [
            {"query_time": query_time}
            | {
                column_name: value
                for column_name, value in zip(
                    self.requested_columns,
                    values_by_column,
                    strict=True,
                )
            }
            for query_time, values_by_column in zip(
                self.query_times, self.mean, strict=True
            )
        ]
        return pandas_module.DataFrame.from_records(rows)


@dataclass(frozen=True, slots=True)
class SampleForecastResult(ForecastResponse):
    """Parsed sampled forecast trajectories with shared metadata and conversions."""

    samples: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for sampled forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            samples=self.samples,
        )

    def to_numpy(self) -> Any:
        """Return NumPy values with axis order ``(sample, horizon, column)``."""
        numpy_module = _require_numpy_module()
        return numpy_module.asarray(self.samples, dtype=float)

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``sample``, ``query_time``, and one value per column."""
        pandas_module = _require_pandas_module()
        rows = [
            {
                "sample": sample_index,
                "query_time": query_time,
                "requested_column": column_name,
                "value": value,
            }
            for sample_index, sample_values in enumerate(self.samples)
            for query_time, values_by_column in zip(
                self.query_times, sample_values, strict=True
            )
            for column_name, value in zip(
                self.requested_columns,
                values_by_column,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per sample and forecast horizon."""
        pandas_module = _require_pandas_module()
        rows = [
            {"sample": sample_index, "query_time": query_time}
            | {
                column_name: value
                for column_name, value in zip(
                    self.requested_columns,
                    values_by_column,
                    strict=True,
                )
            }
            for sample_index, sample_values in enumerate(self.samples)
            for query_time, values_by_column in zip(
                self.query_times, sample_values, strict=True
            )
        ]
        return pandas_module.DataFrame.from_records(rows)


@dataclass(frozen=True, slots=True)
class QuantileForecastResult(ForecastResponse):
    """Parsed quantile forecast surfaces with shared metadata and conversions."""

    quantiles: tuple[QuantileForecast, ...]

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for quantile forecasts."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            quantiles=self.quantiles,
        )

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the ordered quantile levels preserved from the service payload."""
        return tuple(entry.quantile for entry in self.quantiles)

    def to_numpy(self) -> Any:
        """Return NumPy values with axis order ``(quantile, horizon, column)``."""
        numpy_module = _require_numpy_module()
        return numpy_module.asarray(
            [entry.values for entry in self.quantiles],
            dtype=float,
        )

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``quantile``, ``query_time``, and one value per column."""
        pandas_module = _require_pandas_module()
        rows = [
            {
                "quantile": quantile_entry.quantile,
                "query_time": query_time,
                "requested_column": column_name,
                "value": value,
            }
            for quantile_entry in self.quantiles
            for query_time, values_by_column in zip(
                self.query_times,
                quantile_entry.values,
                strict=True,
            )
            for column_name, value in zip(
                self.requested_columns,
                values_by_column,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)

    def to_pandas_wide(self) -> Any:
        """Return a wide DataFrame with one row per quantile and forecast horizon."""
        pandas_module = _require_pandas_module()
        rows = [
            {"quantile": quantile_entry.quantile, "query_time": query_time}
            | {
                column_name: value
                for column_name, value in zip(
                    self.requested_columns,
                    values_by_column,
                    strict=True,
                )
            }
            for quantile_entry in self.quantiles
            for query_time, values_by_column in zip(
                self.query_times,
                quantile_entry.values,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)


@dataclass(frozen=True, slots=True)
class LogProbResult(ForecastResponse):
    """Parsed log densities of observed values, with shared metadata.

    This result carries no forecast values, because a scored request supplies
    them itself as ``query_rows``: what comes back is how plausible the model
    finds them. Each score is joint over the scored columns, so the conversions
    have a horizon axis and no column axis.
    """

    log_prob: LogProbScores

    @property
    def outputs(self) -> ForecastOutputs:
        """Return the legacy nested outputs view for scored requests."""
        return ForecastOutputs(
            query_times=self.query_times,
            requested_columns=self.requested_columns,
            log_prob=self.log_prob,
        )

    def to_numpy(self) -> Any:
        """Return NumPy log densities with axis order ``(horizon,)``."""
        numpy_module = _require_numpy_module()
        return numpy_module.asarray(self.log_prob.values, dtype=float)

    def to_pandas_tidy(self) -> Any:
        """Return a tidy DataFrame with ``query_time``, ``log_prob``, and ``nll``."""
        pandas_module = _require_pandas_module()
        rows = [
            {"query_time": query_time, "log_prob": score, "nll": negated}
            for query_time, score, negated in zip(
                self.query_times,
                self.log_prob.values,
                self.log_prob.nll_values,
                strict=True,
            )
        ]
        return pandas_module.DataFrame.from_records(rows)

    def to_pandas_wide(self) -> Any:
        """Return the one-row request summary over the scored horizons.

        The other result classes widen the column axis, which a log density does
        not have. The wide view is therefore the request-level summary the
        service reports, and the per-horizon scores stay in the tidy view.
        """
        pandas_module = _require_pandas_module()
        return pandas_module.DataFrame.from_records(
            [
                {
                    "total": self.log_prob.total,
                    "mean": self.log_prob.mean,
                    "nll_total": self.log_prob.nll_total,
                    "nll_mean": self.log_prob.nll_mean,
                }
            ]
        )


def build_forecast_payload(
    *,
    model_version: str,
    schema: DataFrameSchema,
    history_rows: Sequence[Mapping[str, Any]],
    query_times: Sequence[Any],
    requested_columns: Sequence[str | int] | None = None,
    return_mode: ReturnMode = "mean",
    n_samples: int | None = None,
    quantiles: Sequence[float | int] | None = None,
    seed: int | None = None,
    schema_version: str = SCHEMA_VERSION,
    query_mode: QueryMode = "forecast",
    condition: ConditionBlock | None = None,
    query_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a validated JSON-compatible forecast request payload."""
    return ForecastRequest(
        metadata=ForecastRequestMetadata(
            model_version=model_version,
            schema_version=schema_version,
            query_mode=query_mode,
            return_mode=return_mode,
        ),
        schema=schema,
        history_rows=history_rows,
        query_times=query_times,
        requested_columns=requested_columns,
        n_samples=n_samples,
        quantiles=quantiles,
        seed=seed,
        condition=condition,
        query_rows=query_rows,
    ).to_payload()


def require_condition_support(
    metadata: HealthMetadata,
    block: ConditionBlock,
) -> None:
    """Refuse a conditioning request the mounted deployment has not advertised.

    The gate exists so a caller learns a deployment cannot condition *before*
    paying for the request, rather than from the error on one it already sent.
    It reads the advertisement alone, which the deployment derives from its own
    head, so it can never claim a capability the checkpoint lacks.

    Raises:
        UnsupportedServiceContractError: when the deployment serves no
            ``condition`` mode, or none of the condition kinds this block uses.
    """
    if "condition" not in metadata.supported_query_modes:
        raise UnsupportedServiceContractError(
            f"The mounted {metadata.head!r} deployment does not serve condition "
            f"queries; it advertises {list(metadata.supported_query_modes)}"
        )
    missing = [
        kind for kind in block.kinds if kind not in metadata.supported_condition_kinds
    ]
    if missing:
        raise UnsupportedServiceContractError(
            f"The mounted {metadata.head!r} deployment does not serve condition "
            f"kinds {missing}; it advertises "
            f"{list(metadata.supported_condition_kinds)}"
        )


def validate_service_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_model_version: str | None = None,
) -> None:
    """Validate `/healthz` metadata against the SDK's compatibility policy.

    Requires the schema version in ``SCHEMA_VERSION``, the expected model
    version when supplied, the advertised query/return/time-index modes,
    and a supported ``decoding_strategy``.
    """
    schema_version = _required_string(metadata, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            "Unsupported JointFM schema_version: "
            f"expected {SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    model_version = _required_string(metadata, "model_version")
    if expected_model_version is not None and model_version != expected_model_version:
        raise UnsupportedModelVersionError(
            "Unsupported JointFM model_version: "
            f"expected {expected_model_version!r}, got {model_version!r}"
        )

    # Query modes are per deployment: the service derives them from the mounted
    # head, so a head that cannot condition advertises fewer modes than this SDK
    # knows. Requiring equality here would reject exactly the deployments the
    # advertisement exists to describe.
    _require_advertised_subset(
        metadata,
        field="supported_query_modes",
        supported_values=SUPPORTED_QUERY_MODES,
        allow_empty=False,
    )
    _require_advertised_subset(
        metadata,
        field="supported_condition_kinds",
        supported_values=SUPPORTED_CONDITION_KINDS,
        allow_empty=True,
    )
    _require_exact_values(
        metadata,
        field="supported_return_modes",
        supported_values=SUPPORTED_RETURN_MODES,
    )
    _require_exact_values(
        metadata,
        field="supported_time_index_modes",
        supported_values=SUPPORTED_TIME_INDEX_MODES,
    )
    _require_decoding_strategy(metadata.get("decoding_strategy"))


def _required_string(metadata: Mapping[str, Any], field: str) -> str:
    """Return a required non-empty string metadata field."""
    value = metadata.get(field)
    if not isinstance(value, str) or value == "":
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} must be a non-empty string"
        )
    return value


def _require_decoding_strategy(value: Any) -> DecodingStrategy:
    """Return a supported horizon decoding strategy from health metadata."""
    if not isinstance(value, str) or value == "":
        raise UnsupportedServiceContractError(
            "JointFM health metadata field 'decoding_strategy' must be a "
            "non-empty string"
        )
    if value not in SUPPORTED_DECODING_STRATEGIES:
        raise UnsupportedServiceContractError(
            "Unsupported JointFM decoding_strategy: expected one of "
            f"{sorted(SUPPORTED_DECODING_STRATEGIES)!r}, got {value!r}"
        )
    return cast(DecodingStrategy, value)


def _require_advertised_subset(
    metadata: Mapping[str, Any],
    *,
    field: str,
    supported_values: Sequence[str],
    allow_empty: bool,
) -> None:
    """Require one advertised capability list to be a subset this SDK understands.

    Used where the deployment legitimately serves fewer values than the SDK
    knows, because the service derives the list from its own checkpoint. An
    advertised value the SDK does not know is still a contract mismatch: it
    means the deployment is newer than this client.
    """
    advertised_values = metadata.get(field)
    if not isinstance(advertised_values, Sequence) or isinstance(
        advertised_values, str | bytes | bytearray
    ):
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} must be a JSON array of strings"
        )

    parsed_values: list[str] = []
    for index, advertised_value in enumerate(advertised_values):
        if not isinstance(advertised_value, str) or advertised_value == "":
            raise UnsupportedServiceContractError(
                f"JointFM health metadata field {field!r}[{index}] must be a non-empty string"
            )
        parsed_values.append(advertised_value)

    if not parsed_values and not allow_empty:
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} advertises nothing"
        )
    unknown = sorted(set(parsed_values) - set(supported_values))
    if unknown:
        raise UnsupportedServiceContractError(
            f"Unsupported JointFM {field}: this client does not know {unknown!r}; "
            f"it supports {sorted(supported_values)!r}"
        )


def _require_exact_values(
    metadata: Mapping[str, Any],
    *,
    field: str,
    supported_values: Sequence[str],
) -> None:
    """Require one advertised capability list to match the SDK's service contract."""
    advertised_values = metadata.get(field)
    if not isinstance(advertised_values, Sequence) or isinstance(
        advertised_values, str | bytes | bytearray
    ):
        raise UnsupportedServiceContractError(
            f"JointFM health metadata field {field!r} must be a JSON array of strings"
        )

    parsed_values: list[str] = []
    for index, advertised_value in enumerate(advertised_values):
        if not isinstance(advertised_value, str) or advertised_value == "":
            raise UnsupportedServiceContractError(
                f"JointFM health metadata field {field!r}[{index}] must be a non-empty string"
            )
        parsed_values.append(advertised_value)

    if set(parsed_values) != set(supported_values):
        raise UnsupportedServiceContractError(
            f"Unsupported JointFM {field}: expected {sorted(supported_values)!r}, "
            f"got {sorted(parsed_values)!r}"
        )


def _validate_time_value_options(column_spec: ColumnSpec) -> None:
    uses_time_value_options = any(
        [
            column_spec.time_value_kind is not None,
            column_spec.time_value_scale_seconds is not None,
            column_spec.time_value_use_local_normalized_time,
            column_spec.time_value_calendar_id != DEFAULT_CALENDAR_ID,
            column_spec.time_value_timezone is not None,
        ]
    )
    if column_spec.modality != "time_value":
        if uses_time_value_options:
            raise ValueError("time_value options require modality='time_value'")
        return
    if column_spec.time_value_kind != "absolute_datetime" and any(
        [
            column_spec.time_value_scale_seconds is not None,
            column_spec.time_value_use_local_normalized_time,
            column_spec.time_value_calendar_id != DEFAULT_CALENDAR_ID,
            column_spec.time_value_timezone is not None,
        ]
    ):
        raise ValueError(
            "time_value absolute datetime options require time_value_kind='absolute_datetime'"
        )


def _validate_history_declared_columns(
    history_rows: Sequence[Mapping[str, Any]],
    schema: DataFrameSchema,
) -> None:
    present_columns: set[str] = set()
    for history_row in history_rows:
        present_columns.update(history_row.keys())
    expected_columns = {column.name for column in schema.columns}
    missing_columns = sorted(expected_columns.difference(present_columns))
    if missing_columns:
        raise ValueError(
            f"history_rows are missing declared columns: {missing_columns}"
        )
    if schema.time_column is not None and schema.time_column not in present_columns:
        raise ValueError(f"history_rows are missing time_column {schema.time_column!r}")


def _serialize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    return [
        {
            key: _to_json_compatible(value, field=f"{field}[{index}].{key}")
            for key, value in row.items()
        }
        for index, row in enumerate(rows)
    ]


def _require_scored_value(value: Any, *, field: str) -> None:
    if value is None:
        raise ValueError(f"{field} must carry an observed value to score")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _serialize_query_times(
    query_times: Sequence[Any],
    *,
    time_index_mode: TimeIndexMode,
) -> list[Any]:
    values = _require_sequence(query_times, field="query_times")
    if time_index_mode == "absolute_datetime":
        return [
            _serialize_absolute_datetime(value, field=f"query_times[{index}]")
            for index, value in enumerate(values)
        ]
    if time_index_mode == "ordinal":
        serialized: list[int] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"query_times[{index}] must be an integer for ordinal mode"
                )
            serialized.append(value)
        return serialized

    serialized_float_times: list[int | float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"query_times[{index}] must be numeric for continuous_float mode"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"query_times[{index}] must be finite")
        serialized_float_times.append(value)
    return serialized_float_times


def _serialize_absolute_datetime(value: Any, *, field: str) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        normalized_value = (
            value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
        )
        try:
            timestamp = datetime.fromisoformat(normalized_value)
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO 8601 datetime") from error
    else:
        raise ValueError(
            f"{field} must be a timezone-aware datetime or ISO 8601 string"
        )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must include timezone information")
    return timestamp.astimezone(timezone.utc).isoformat()


def _resolve_requested_columns(
    column_specs: Sequence[ColumnSpec],
    requested_columns: Sequence[str | int] | None,
) -> list[str] | None:
    if requested_columns is None:
        return None
    requested_values = _require_sequence(requested_columns, field="requested_columns")
    column_names = [column.name for column in column_specs]
    resolved_names: list[str] = []
    for index, requested_column in enumerate(requested_values):
        if isinstance(requested_column, str):
            if requested_column not in column_names:
                raise ValueError(
                    f"requested_columns[{index}] references unknown column {requested_column!r}"
                )
            resolved_names.append(requested_column)
            continue
        if isinstance(requested_column, bool) or not isinstance(requested_column, int):
            raise ValueError(
                f"requested_columns[{index}] must be a string name or integer column index"
            )
        if requested_column < 0 or requested_column >= len(column_names):
            raise ValueError(
                f"requested_columns[{index}] index {requested_column} is out of bounds"
            )
        resolved_names.append(column_names[requested_column])
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("requested_columns must not contain duplicates")
    return resolved_names


def _require_quantiles(value: Sequence[float | int] | None) -> list[float]:
    quantiles = _require_sequence(value, field="quantiles")
    parsed: list[float] = []
    for index, quantile in enumerate(quantiles):
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise ValueError(f"quantiles[{index}] must be numeric")
        parsed_quantile = float(quantile)
        if parsed_quantile <= 0.0 or parsed_quantile >= 1.0:
            raise ValueError(f"quantiles[{index}] must lie strictly between 0 and 1")
        parsed.append(parsed_quantile)
    return parsed


def _structured_error_tuple(value: Any) -> tuple[StructuredError, ...]:
    if value is None:
        return ()
    errors = _require_sequence(value, field="errors", allow_empty=True)
    parsed_errors: list[StructuredError] = []
    for index, error in enumerate(errors):
        parsed_errors.append(
            StructuredError.from_payload(
                _require_mapping(error, field=f"errors[{index}]")
            )
        )
    return tuple(parsed_errors)


def _string_tuple(
    value: Any, *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _require_sequence(value, field=field, allow_empty=allow_empty)
    return tuple(_require_string(item, field=f"{field}[]") for item in values)


def _optional_tuple(value: Any, *, field: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(_require_sequence(value, field=field, allow_empty=True))


@dataclass(frozen=True, slots=True)
class _ForecastResponseExpectations:
    schema_version: str | None = None
    model_version: str | None = None
    query_mode: QueryMode | None = None
    return_mode: ReturnMode | None = None
    query_times: tuple[Any, ...] | None = None
    horizon_count: int | None = None
    requested_columns: tuple[str, ...] | None = None
    requested_column_count: int | None = None
    n_samples: int | None = None
    quantiles: tuple[float, ...] | None = None


def _forecast_response_expectations(
    request_payload: Mapping[str, Any] | None,
) -> _ForecastResponseExpectations:
    if request_payload is None:
        return _ForecastResponseExpectations()
    if not isinstance(request_payload, Mapping):
        raise ValueError("request_payload must be a JSON object")

    query_times_value = request_payload.get("query_times")
    query_times = (
        tuple(_require_sequence(query_times_value, field="request_payload.query_times"))
        if query_times_value is not None
        else None
    )
    requested_columns: tuple[str, ...] | None = None
    requested_column_count: int | None = None
    requested_columns_value = request_payload.get("requested_columns")
    if requested_columns_value is not None:
        requested_column_values = _require_sequence(
            requested_columns_value,
            field="request_payload.requested_columns",
        )
        requested_column_count = len(requested_column_values)
        if all(
            isinstance(value, str) and value != "" for value in requested_column_values
        ):
            requested_columns = tuple(requested_column_values)

    condition_value = request_payload.get("condition")
    if condition_value is not None and query_times is not None:
        condition_block = _require_mapping(
            condition_value, field="request_payload.condition"
        )
        conditioned_index = condition_block.get("query_time_index")
        if isinstance(conditioned_index, int) and not isinstance(
            conditioned_index, bool
        ):
            if 0 <= conditioned_index < len(query_times):
                query_times = (query_times[conditioned_index],)

    query_mode_value = request_payload.get("query_mode")
    return_mode_value = request_payload.get("return_mode")
    quantiles_value = request_payload.get("quantiles")
    return _ForecastResponseExpectations(
        schema_version=_optional_string(
            request_payload.get("schema_version"),
            field="request_payload.schema_version",
        ),
        model_version=_optional_string(
            request_payload.get("model_version"),
            field="request_payload.model_version",
        ),
        query_mode=None
        if query_mode_value is None
        else _require_member(
            query_mode_value,
            field="request_payload.query_mode",
            supported_values=SUPPORTED_QUERY_MODES,
        ),
        return_mode=None
        if return_mode_value is None
        else _require_member(
            return_mode_value,
            field="request_payload.return_mode",
            supported_values=SUPPORTED_RETURN_MODES,
        ),
        query_times=query_times,
        horizon_count=None if query_times is None else len(query_times),
        requested_columns=requested_columns,
        requested_column_count=requested_column_count,
        n_samples=_optional_positive_int(
            request_payload.get("n_samples"),
            field="request_payload.n_samples",
        ),
        quantiles=None
        if quantiles_value is None
        else tuple(_require_quantiles(quantiles_value)),
    )


def _validate_sample_bounds(
    samples: tuple[tuple[tuple[float, ...], ...], ...],
    requested_columns: Sequence[str],
    request_payload: Mapping[str, Any] | None,
) -> None:
    if request_payload is None:
        return

    bounds_by_column = _sample_bounds_by_column_name(request_payload)
    if not bounds_by_column:
        return

    for requested_column_index, requested_column in enumerate(requested_columns):
        bounds = bounds_by_column.get(requested_column)
        if bounds is None:
            continue
        lower_bound, upper_bound = bounds
        for sample_index, sample_values in enumerate(samples):
            for horizon_index, horizon_values in enumerate(sample_values):
                value = horizon_values[requested_column_index]
                field = (
                    f"outputs.samples[{sample_index}][{horizon_index}]"
                    f"[{requested_column_index}]"
                )
                if lower_bound is not None and value < lower_bound:
                    raise ValueError(
                        f"{field} violates requested lower_bound for column "
                        f"{requested_column!r}: {value} < {lower_bound}"
                    )
                if upper_bound is not None and value > upper_bound:
                    raise ValueError(
                        f"{field} violates requested upper_bound for column "
                        f"{requested_column!r}: {value} > {upper_bound}"
                    )


def _sample_bounds_by_column_name(
    request_payload: Mapping[str, Any],
) -> dict[str, tuple[float | None, float | None]]:
    columns_value = request_payload.get("columns")
    if columns_value is None:
        return {}

    columns = _require_sequence(columns_value, field="request_payload.columns")
    bounds_by_column: dict[str, tuple[float | None, float | None]] = {}
    for column_index, column_value in enumerate(columns):
        field = f"request_payload.columns[{column_index}]"
        column = _require_mapping(column_value, field=field)
        column_name = _require_string(column.get("name"), field=f"{field}.name")
        lower_bound = _optional_float(
            column.get("lower_bound"),
            field=f"{field}.lower_bound",
        )
        upper_bound = _optional_float(
            column.get("upper_bound"),
            field=f"{field}.upper_bound",
        )
        if lower_bound is None and upper_bound is None:
            continue
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError(
                f"{field}.lower_bound must be less than or equal to upper_bound"
            )
        bounds_by_column[column_name] = (lower_bound, upper_bound)
    return bounds_by_column


def _raise_for_response_errors(payload: Mapping[str, Any]) -> None:
    errors = _structured_error_tuple(payload.get("errors"))
    if not errors:
        return
    error_payloads = tuple(_structured_error_payload(error) for error in errors)
    first_error = errors[0]
    message = f"JointFM response returned {first_error.code}: {first_error.message}"
    raise JointFMServiceError(message, jointfm_errors=error_payloads)


def _structured_error_payload(error: StructuredError) -> dict[str, str]:
    payload = {
        "code": error.code,
        "message": error.message,
    }
    if error.field is not None:
        payload["field"] = error.field
    return payload


def _require_none(value: Any, *, field: str) -> None:
    if value is not None:
        raise ValueError(f"{field} must be null")


def _validate_length(
    *,
    actual_length: int,
    expected_length: int | None,
    field: str,
) -> None:
    if expected_length is None:
        return
    if actual_length != expected_length:
        raise ValueError(
            f"{field} length mismatch: expected {expected_length}, got {actual_length}"
        )


def _require_output_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _require_float_vector(
    value: Any,
    *,
    field: str,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    values = _require_sequence(value, field=field)
    _validate_length(
        actual_length=len(values),
        expected_length=expected_length,
        field=field,
    )
    return tuple(
        _require_output_float(item, field=f"{field}[{index}]")
        for index, item in enumerate(values)
    )


def _require_derived_score(value: float, *, expected: float, field: str) -> float:
    if not math.isclose(
        value,
        expected,
        rel_tol=DERIVED_SCORE_TOLERANCE,
        abs_tol=DERIVED_SCORE_TOLERANCE,
    ):
        raise ValueError(
            f"{field} disagrees with outputs.log_prob.values: "
            f"expected {expected}, got {value}"
        )
    return value


def _require_float_matrix(
    value: Any,
    *,
    field: str,
    expected_outer_length: int | None = None,
    expected_inner_length: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    rows = _require_sequence(value, field=field)
    _validate_length(
        actual_length=len(rows),
        expected_length=expected_outer_length,
        field=field,
    )
    parsed_rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        row_values = _require_sequence(row, field=f"{field}[{row_index}]")
        _validate_length(
            actual_length=len(row_values),
            expected_length=expected_inner_length,
            field=f"{field}[{row_index}]",
        )
        parsed_rows.append(
            tuple(
                _require_output_float(
                    item,
                    field=f"{field}[{row_index}][{column_index}]",
                )
                for column_index, item in enumerate(row_values)
            )
        )
    return tuple(parsed_rows)


def _require_float_tensor3(
    value: Any,
    *,
    field: str,
    expected_outer_length: int | None = None,
    expected_middle_length: int | None = None,
    expected_inner_length: int | None = None,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    items = _require_sequence(value, field=field)
    _validate_length(
        actual_length=len(items),
        expected_length=expected_outer_length,
        field=field,
    )
    parsed_items: list[tuple[tuple[float, ...], ...]] = []
    for item_index, item in enumerate(items):
        parsed_items.append(
            _require_float_matrix(
                item,
                field=f"{field}[{item_index}]",
                expected_outer_length=expected_middle_length,
                expected_inner_length=expected_inner_length,
            )
        )
    return tuple(parsed_items)


def _require_quantile_forecasts(
    value: Any,
    *,
    field: str,
    expected_quantiles: Sequence[float] | None = None,
    expected_horizon_count: int | None = None,
    expected_column_count: int | None = None,
) -> tuple[QuantileForecast, ...]:
    quantile_entries = _require_sequence(value, field=field)
    _validate_length(
        actual_length=len(quantile_entries),
        expected_length=None if expected_quantiles is None else len(expected_quantiles),
        field=field,
    )
    parsed_quantiles: list[QuantileForecast] = []
    for quantile_index, entry in enumerate(quantile_entries):
        mapping = _require_mapping(entry, field=f"{field}[{quantile_index}]")
        quantile = _require_output_float(
            mapping.get("quantile"),
            field=f"{field}[{quantile_index}].quantile",
        )
        if quantile <= 0.0 or quantile >= 1.0:
            raise ValueError(
                f"{field}[{quantile_index}].quantile must lie strictly between 0 and 1"
            )
        parsed_quantiles.append(
            QuantileForecast(
                quantile=quantile,
                values=_require_float_matrix(
                    mapping.get("values"),
                    field=f"{field}[{quantile_index}].values",
                    expected_outer_length=expected_horizon_count,
                    expected_inner_length=expected_column_count,
                ),
            )
        )
    if expected_quantiles is not None and tuple(
        entry.quantile for entry in parsed_quantiles
    ) != tuple(expected_quantiles):
        raise ValueError(
            "forecast response quantiles mismatch: "
            f"expected {list(expected_quantiles)!r}, "
            f"got {[entry.quantile for entry in parsed_quantiles]!r}"
        )
    return tuple(parsed_quantiles)


def _require_pandas_module() -> Any:
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "pandas result conversion requires installing jointfm-client[notebooks]"
        ) from error
    return pandas_module


def _require_numpy_module() -> Any:
    try:
        import numpy as numpy_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "NumPy result conversion requires installing jointfm-client[notebooks]"
        ) from error
    return numpy_module


def _to_json_compatible(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, datetime):
        return _serialize_absolute_datetime(value, field=field)
    if isinstance(value, Mapping):
        return {
            _require_string(key, field=f"{field}.key"): _to_json_compatible(
                nested_value,
                field=f"{field}.{key}",
            )
            for key, nested_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_json_compatible(nested_value, field=f"{field}[]")
            for nested_value in value
        ]
    raise ValueError(f"{field} must be JSON-compatible")


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_sequence(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a JSON array")
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{field} must not be empty")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field)


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _require_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return parsed


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field=field)


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _optional_positive_float(value: Any, *, field: str) -> float | None:
    parsed = _optional_float(value, field=field)
    if parsed is not None and parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _optional_mapping(value: Any, *, field: str) -> Mapping[str | int, int] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, field=field)
    parsed: dict[str | int, int] = {}
    for key, target in mapping.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise ValueError(f"{field} keys must be strings or integers")
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError(f"{field} values must be integers")
        parsed[key] = target
    return parsed


_MemberT = TypeVar("_MemberT", bound=str)


def _require_member(
    value: Any, *, field: str, supported_values: Sequence[_MemberT]
) -> _MemberT:
    parsed = _require_string(value, field=field)
    if parsed not in supported_values:
        raise ValueError(
            f"Unsupported {field}: expected one of {tuple(supported_values)!r}, got {parsed!r}"
        )
    return cast(_MemberT, parsed)


def _optional_member(
    value: Any,
    *,
    field: str,
    supported_values: Sequence[_MemberT],
) -> _MemberT | None:
    if value is None:
        return None
    return _require_member(value, field=field, supported_values=supported_values)
