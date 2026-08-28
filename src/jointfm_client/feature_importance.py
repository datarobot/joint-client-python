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

"""Permutation feature importance helpers for ``JointFMClient.feature_importance``.

Each feature column is shuffled across history rows while its marginal is
preserved, and the permuted forecast is compared against one shared baseline
forecast. Two scores come out of that comparison: the absolute shift in
forecast mean, and the centered squared 2-Wasserstein distance between the
baseline and permuted sample sets (which also catches shape changes a mean
shift alone would miss).
"""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import Any


def sample_w2_distance(
    baseline: Sequence[float],
    permuted: Sequence[float],
    *,
    location_invariant: bool = True,
) -> float:
    """Centered squared 2-Wasserstein distance between two 1D sample sets.

    Both sample vectors are one-dimensional draws at the same target and
    horizon, so the distance has a closed form: sort both vectors and take the
    mean squared gap (halved to match the usual W2 cost convention). When
    ``location_invariant`` is true, both vectors are centered first so the
    score reflects spread, skew, and tail movement rather than repeating the
    mean-shift readout.
    """
    if len(baseline) == 0 or len(permuted) == 0:
        return 0.0
    if len(baseline) != len(permuted):
        raise ValueError(
            f"sample vectors must have equal length; got {len(baseline)} and {len(permuted)}"
        )

    left = list(baseline)
    right = list(permuted)
    if location_invariant:
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left = [value - left_mean for value in left]
        right = [value - right_mean for value in right]

    left.sort()
    right.sort()
    squared_gaps = sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )
    return squared_gaps / len(left) / 2.0


def _is_history_row_sequence(history: Any) -> bool:
    """Return whether ``history`` is a sequence of row mappings, not a DataFrame."""
    return isinstance(history, Sequence) and not isinstance(
        history, str | bytes | bytearray
    )


def permute_history_column(history: Any, feature: str, *, seed: int) -> Any:
    """Return a copy of ``history`` with ``feature`` shuffled across rows."""
    rng = random.Random(seed)
    if _is_history_row_sequence(history):
        rows = [dict(row) for row in history]
        present_indices = [index for index, row in enumerate(rows) if feature in row]
        if not present_indices:
            raise ValueError(f"history rows are missing column {feature!r}")
        values = [rows[index][feature] for index in present_indices]
        rng.shuffle(values)
        for index, value in zip(present_indices, values, strict=True):
            rows[index][feature] = value
        return rows

    pandas_module = _require_pandas_module()
    if not isinstance(history, pandas_module.DataFrame):
        raise ValueError(
            "history must be a pandas DataFrame or a sequence of row mappings"
        )
    if feature not in history.columns:
        raise ValueError(f"history frame is missing column {feature!r}")
    permuted = history.copy()
    values = list(history[feature])
    rng.shuffle(values)
    permuted[feature] = values
    return permuted


def feature_importance_entry(
    *,
    feature: str,
    horizons: Sequence[int],
    target_columns: Sequence[str],
    baseline_samples: Sequence[Sequence[Sequence[float]]],
    permuted_samples: Sequence[Sequence[Sequence[float]]],
    baseline_columns: Sequence[str],
) -> dict[str, Any]:
    """Score one permuted feature against the shared baseline forecast.

    Returns ``{"feature": ..., "mean": {target: {horizon: value}}, "distance":
    {target: {horizon: value}}}``, where ``mean`` is the absolute shift in
    forecast mean and ``distance`` is ``sample_w2_distance``, both indexed by
    every requested target and horizon.
    """
    mean_scores: dict[str, dict[int, float]] = {}
    distance_scores: dict[str, dict[int, float]] = {}
    for target in target_columns:
        target_index = list(baseline_columns).index(target)
        mean_by_horizon: dict[int, float] = {}
        distance_by_horizon: dict[int, float] = {}
        for horizon_index, horizon in enumerate(horizons):
            baseline_values = [
                sample[horizon_index][target_index] for sample in baseline_samples
            ]
            permuted_values = [
                sample[horizon_index][target_index] for sample in permuted_samples
            ]
            baseline_mean = sum(baseline_values) / len(baseline_values)
            permuted_mean = sum(permuted_values) / len(permuted_values)
            mean_by_horizon[horizon] = abs(permuted_mean - baseline_mean)
            distance_by_horizon[horizon] = sample_w2_distance(
                baseline_values, permuted_values
            )
        mean_scores[target] = mean_by_horizon
        distance_scores[target] = distance_by_horizon
    return {"feature": feature, "mean": mean_scores, "distance": distance_scores}


def _require_pandas_module() -> Any:
    """Return pandas or raise with the SDK extra needed for DataFrame history."""
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "pandas history support requires installing jointfm-client[notebooks]"
        ) from error
    return pandas_module


__all__ = [
    "feature_importance_entry",
    "permute_history_column",
    "sample_w2_distance",
]
