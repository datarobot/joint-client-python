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

"""Tests for permutation feature importance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jointfm_client import ColumnSpec, JointFMClient
from jointfm_client.feature_importance import (
    permute_history_column,
    sample_w2_distance,
)

_MODEL_VERSION = "jointfm-inference:0.2.0+ckpt.sdk-test"


class _ImportanceTransport:
    """Returns one queued sample response per `post_json` call, in order."""

    def __init__(self, sample_batches: list[list[list[list[float]]]]) -> None:
        """Init."""
        self.sample_batches = sample_batches
        self.payloads: list[Mapping[str, Any]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        raise AssertionError(f"unexpected health request: {url}")

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        self.payloads.append(dict(payload))
        samples = self.sample_batches[len(self.payloads) - 1]
        return {
            "schema_version": "v1",
            "image_version": "0.2.0",
            "model_version": _MODEL_VERSION,
            "checkpoint_version": "sdk-test",
            "head": "studentt",
            "query_mode": "forecast",
            "return_mode": "samples",
            "outputs": {
                "query_times": [2, 3],
                "requested_columns": ["target"],
                "mean": None,
                "samples": samples,
                "quantiles": None,
            },
            "diagnostics": {
                "history_rows": 2,
                "horizon_count": 2,
                "seed": payload.get("seed"),
            },
            "errors": [],
        }


def test_feature_importance_scores_mean_shift_and_distribution_distance() -> None:
    """One feature's mean shift and distance are scored per target and horizon."""
    transport = _ImportanceTransport(
        sample_batches=[
            [[[10.0], [20.0]], [[12.0], [22.0]]],
            [[[11.0], [15.0]], [[13.0], [27.0]]],
        ]
    )
    client = JointFMClient(
        predict_url="http://localhost:8080/predict",
        transport=transport,
    )
    columns = (
        ColumnSpec(name="target", modality="numeric", role="target"),
        ColumnSpec(name="feat", modality="numeric", role="feature"),
    )
    history = [{"target": 10.0, "feat": 1.0}, {"target": 11.0, "feat": 2.0}]

    result = client.feature_importance(
        history,
        query_times=[2, 3],
        horizons=[1, 2],
        feature_columns=["feat"],
        target_columns=["target"],
        columns=columns,
        model_version=_MODEL_VERSION,
        n_samples=2,
        seed=7,
    )

    # horizon 1: means 11.0 -> 12.0 (shift 1.0), identical shape once centered (distance 0).
    # horizon 2: means 21.0 -> 21.0 (no shift), spread widens from +-1 to +-6 (distance 12.5).
    assert result == [
        {
            "feature": "feat",
            "mean": {"target": {1: 1.0, 2: 0.0}},
            "distance": {"target": {1: 0.0, 2: 12.5}},
        }
    ]
    assert len(transport.payloads) == 2
    assert [payload["seed"] for payload in transport.payloads] == [7, 7]

    baseline_feature_values = [
        row["feat"] for row in transport.payloads[0]["history_rows"]
    ]
    permuted_feature_values = [
        row["feat"] for row in transport.payloads[1]["history_rows"]
    ]
    assert baseline_feature_values == [1.0, 2.0]
    assert sorted(permuted_feature_values) == sorted(baseline_feature_values)


def test_feature_importance_requires_non_empty_feature_columns() -> None:
    """feature_importance rejects an empty feature_columns list."""
    client = JointFMClient(predict_url="http://localhost:8080/predict")

    with pytest.raises(ValueError, match="feature_columns must not be empty"):
        client.feature_importance(
            [{"target": 10.0}],
            query_times=[2],
            horizons=[1],
            feature_columns=[],
            target_columns=["target"],
            model_version=_MODEL_VERSION,
        )


def test_feature_importance_requires_non_empty_target_columns() -> None:
    """feature_importance rejects an empty target_columns list."""
    client = JointFMClient(predict_url="http://localhost:8080/predict")

    with pytest.raises(ValueError, match="target_columns must not be empty"):
        client.feature_importance(
            [{"target": 10.0, "feat": 1.0}],
            query_times=[2],
            horizons=[1],
            feature_columns=["feat"],
            target_columns=[],
            model_version=_MODEL_VERSION,
        )


def test_feature_importance_requires_horizons_matching_query_times_length() -> None:
    """feature_importance rejects a horizons/query_times length mismatch."""
    client = JointFMClient(predict_url="http://localhost:8080/predict")

    with pytest.raises(ValueError, match="horizons must have the same length"):
        client.feature_importance(
            [{"target": 10.0, "feat": 1.0}],
            query_times=[2, 3],
            horizons=[1],
            feature_columns=["feat"],
            target_columns=["target"],
            model_version=_MODEL_VERSION,
        )


def test_permute_history_column_shuffles_row_sequence_values() -> None:
    """permute_history_column reshuffles one column across row mappings."""
    history = [{"feat": 1.0, "target": 10.0}, {"feat": 2.0, "target": 11.0}]

    permuted = permute_history_column(history, "feat", seed=1)

    assert sorted(row["feat"] for row in permuted) == [1.0, 2.0]
    assert [row["target"] for row in permuted] == [10.0, 11.0]
    # original history is left untouched
    assert history == [{"feat": 1.0, "target": 10.0}, {"feat": 2.0, "target": 11.0}]


def test_permute_history_column_shuffles_dataframe_values() -> None:
    """permute_history_column reshuffles one column across DataFrame rows."""
    pandas_module = pytest.importorskip("pandas")
    frame = pandas_module.DataFrame({"feat": [1.0, 2.0], "target": [10.0, 11.0]})

    permuted = permute_history_column(frame, "feat", seed=1)

    assert sorted(permuted["feat"].tolist()) == [1.0, 2.0]
    assert permuted["target"].tolist() == [10.0, 11.0]
    assert frame["feat"].tolist() == [1.0, 2.0]


def test_permute_history_column_rejects_missing_column() -> None:
    """permute_history_column raises when the feature column is absent."""
    with pytest.raises(ValueError, match="missing column"):
        permute_history_column([{"target": 10.0}], "feat", seed=1)


def test_permute_history_column_rejects_unsupported_history_type() -> None:
    """permute_history_column raises for a history that is neither rows nor a frame."""
    with pytest.raises(ValueError, match="pandas DataFrame or a sequence"):
        permute_history_column({"target": 10.0}, "feat", seed=1)


def test_permute_history_column_rejects_dataframe_missing_column() -> None:
    """permute_history_column raises when the DataFrame lacks the feature column."""
    pandas_module = pytest.importorskip("pandas")
    frame = pandas_module.DataFrame({"target": [10.0, 11.0]})

    with pytest.raises(ValueError, match="missing column"):
        permute_history_column(frame, "feat", seed=1)


def test_sample_w2_distance_is_zero_for_identically_shaped_shift() -> None:
    """A pure mean shift scores zero distance once centered."""
    assert sample_w2_distance([10.0, 12.0], [11.0, 13.0]) == 0.0


def test_sample_w2_distance_rejects_mismatched_lengths() -> None:
    """sample_w2_distance requires equal-length sample vectors."""
    with pytest.raises(ValueError, match="equal length"):
        sample_w2_distance([1.0, 2.0], [1.0])


def test_sample_w2_distance_returns_zero_for_empty_inputs() -> None:
    """sample_w2_distance returns zero rather than dividing by zero."""
    assert sample_w2_distance([], []) == 0.0
