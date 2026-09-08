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

"""Capability-aware forecast planning built from `/healthz` metadata.

The planner checks a notebook's desired ``(feature_columns, target_columns)``
split against the data-generation envelope advertised by the deployment. That
envelope caps the *total* number of series in one request rather than the two
roles separately, so the planner validates the combined width against
``min_series`` and ``max_series``, and the request shape against ``n_input``
and ``n_output``, before anything is sent — callers then see a clear, local
exception instead of a service-side validation error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jointfm_client.contract import ColumnSpec, HealthMetadata
from jointfm_client.exceptions import JointFMCapacityError


@dataclass(frozen=True, slots=True)
class ForecastPlan:
    """Validated forecast request layout for one deployment.

    ``feature_columns`` and ``target_columns`` are the caller's split, carried
    through unchanged — the deployment budgets the two together, so neither
    role is ever rewritten to fit. ``columns`` is the matching ordered
    ``ColumnSpec`` list and ``requested_columns`` is the caller's target list,
    since callers consistently want predictions for the targets they declared.
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

    Raises :class:`JointFMCapacityError` when the combined column count falls
    outside ``[min_series, max_series]``, when the caller passed no target
    column, when feature and target names overlap, or when the history length
    or horizon count exceeds capacity. ``n_input`` and ``n_output`` are the
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
    feature_list: list[str] = list(feature_columns)
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

    _check_series_capacity(
        feature_list,
        target_list,
        min_series=capacity.min_series,
        max_series=capacity.max_series,
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
        ColumnSpec(name=name, modality="numeric", role="target") for name in target_list
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


def _check_series_capacity(
    feature_list: Sequence[str],
    target_list: Sequence[str],
    *,
    min_series: int,
    max_series: int,
) -> None:
    """Check the combined feature and target width against the series budget.

    The deployment caps features and targets together, so the two lists are
    reported jointly: a caller over budget needs to know which columns made up
    the total, not merely that one role was too wide.
    """
    series_count = len(feature_list) + len(target_list)
    if series_count > max_series:
        raise JointFMCapacityError(
            f"Deployment supports at most max_series={max_series} series per "
            f"request, counting features and targets together; got "
            f"{series_count} ({len(feature_list)} feature(s) "
            f"{list(feature_list)!r} + {len(target_list)} target(s) "
            f"{list(target_list)!r}). Drop columns to fit the budget."
        )
    if series_count < min_series:
        raise JointFMCapacityError(
            f"Deployment requires at least min_series={min_series} series per "
            f"request, counting features and targets together; got "
            f"{series_count} ({len(feature_list)} feature(s) "
            f"{list(feature_list)!r} + {len(target_list)} target(s) "
            f"{list(target_list)!r})."
        )


__all__ = ["ForecastPlan", "plan_forecast_columns"]
