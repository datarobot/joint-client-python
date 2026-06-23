"""Capability-aware forecast planning built from `/healthz` metadata.

The planner adapts a notebook's desired ``(feature_columns, target_columns)``
split to the data-generation envelope advertised by the deployment. A
deployment that only supports targets (``max_features == 0``) needs every
would-be feature passed as a target column; deployments with capacity caps
need explicit checks against ``max_features``, ``max_targets``, ``n_input``,
and ``n_output`` before a request is sent so callers see a clear, local
exception instead of a service-side validation error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

from jointfm_client.contract import ColumnSpec, HealthMetadata
from jointfm_client.exceptions import JointFMCapacityError

_LOGGER = logging.getLogger("jointfm_client.capabilities")


@dataclass(frozen=True, slots=True)
class ForecastPlan:
    """Validated forecast request layout for one deployment.

    ``feature_columns`` and ``target_columns`` reflect the column split after
    any feature-to-target downgrade applied for ``max_features == 0``
    deployments. ``columns`` is the matching ordered ``ColumnSpec`` list and
    ``requested_columns`` is the caller's original target list, since callers
    consistently want predictions for the conceptual targets they declared.
    """

    columns: tuple[ColumnSpec, ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    requested_columns: tuple[str, ...]


def plan_forecast_columns(
    *,
    health: HealthMetadata,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    history_length: int,
    query_times_length: int,
) -> ForecastPlan:
    """Plan a forecast request that fits the deployment's data-generation envelope.

    Downgrades all ``feature_columns`` to target columns when the deployment
    advertises ``max_features == 0`` and emits a warning on
    ``jointfm_client.capabilities``. Raises :class:`JointFMCapacityError` when
    the resulting column split, history length, or horizon count exceeds the
    deployment's capacity. ``n_input`` and ``n_output`` are interpreted as the
    largest history window and forecast horizon the deployed model was trained
    to handle; smaller requests are allowed.
    """
    capacity = health.data_generation
    if capacity is None:
        raise JointFMCapacityError(
            "JointFM deployment health metadata is missing the 'data_generation' "
            "block; cannot plan a forecast request against an unknown capacity "
            "envelope."
        )

    requested_columns = tuple(target_columns)
    original_feature_list = list(feature_columns)
    feature_list: list[str] = list(original_feature_list)
    target_list: list[str] = list(target_columns)
    if not target_list:
        raise JointFMCapacityError(
            "plan_forecast_columns requires at least one target column"
        )

    duplicate_names = _find_duplicates(feature_list + target_list)
    if duplicate_names:
        raise JointFMCapacityError(
            f"feature_columns and target_columns must be disjoint and unique; "
            f"duplicates: {duplicate_names!r}"
        )

    if capacity.max_features == 0 and feature_list:
        _LOGGER.warning(
            "Deployment advertises max_features=0; downgrading %d feature "
            "column(s) to target columns: %s",
            len(feature_list),
            feature_list,
        )
        target_list = feature_list + target_list
        feature_list = []

    _check_column_capacity(
        feature_list,
        target_list,
        capacity_min=capacity.min_features,
        capacity_max=capacity.max_features,
        target_min=capacity.min_targets,
        target_max=capacity.max_targets,
    )

    if history_length <= 0:
        raise JointFMCapacityError("history_length must be positive")
    if history_length > capacity.n_input:
        raise JointFMCapacityError(
            f"Deployment was trained on at most n_input={capacity.n_input} history "
            f"rows; got {history_length}. Trim the history to fit the deployed "
            f"model's training window."
        )
    if query_times_length <= 0:
        raise JointFMCapacityError("query_times_length must be positive")
    if query_times_length > capacity.n_output:
        raise JointFMCapacityError(
            f"Deployment supports at most n_output={capacity.n_output} query "
            f"times per request; got {query_times_length}."
        )

    columns = tuple(
        ColumnSpec(name=name, modality="numeric", role="feature")
        for name in feature_list
    ) + tuple(
        ColumnSpec(name=name, modality="numeric", role="target")
        for name in target_list
    )

    final_names = {column.name for column in columns}
    missing_requested = [name for name in requested_columns if name not in final_names]
    if missing_requested:
        raise JointFMCapacityError(
            f"requested_columns refer to columns not in the planned schema: "
            f"{missing_requested!r}"
        )

    return ForecastPlan(
        columns=columns,
        feature_columns=tuple(feature_list),
        target_columns=tuple(target_list),
        requested_columns=requested_columns,
    )


def _find_duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _check_column_capacity(
    feature_list: Sequence[str],
    target_list: Sequence[str],
    *,
    capacity_min: int,
    capacity_max: int,
    target_min: int,
    target_max: int,
) -> None:
    if len(feature_list) > capacity_max:
        raise JointFMCapacityError(
            f"Deployment supports at most max_features={capacity_max} feature "
            f"column(s); got {len(feature_list)}: {list(feature_list)!r}"
        )
    if len(feature_list) < capacity_min:
        raise JointFMCapacityError(
            f"Deployment requires at least min_features={capacity_min} feature "
            f"column(s); got {len(feature_list)}: {list(feature_list)!r}"
        )
    if len(target_list) > target_max:
        raise JointFMCapacityError(
            f"Deployment supports at most max_targets={target_max} target "
            f"column(s); got {len(target_list)}: {list(target_list)!r}"
        )
    if len(target_list) < target_min:
        raise JointFMCapacityError(
            f"Deployment requires at least min_targets={target_min} target "
            f"column(s); got {len(target_list)}: {list(target_list)!r}"
        )


__all__ = ["ForecastPlan", "plan_forecast_columns"]
