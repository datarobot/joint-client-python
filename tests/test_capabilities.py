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

"""Tests for the capabilities surface of jointfm_client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from jointfm_client import (
    ColumnSpec,
    DataGenerationCapabilities,
    ForecastPlan,
    HealthMetadata,
    JointFMCapacityError,
    plan_forecast_columns,
)


def _capabilities(**overrides: Any) -> DataGenerationCapabilities:
    """Build a permissive capabilities object that tests override per case."""
    defaults: dict[str, Any] = {
        "sampler_type": "studentt",
        "min_series": 1,
        "max_series": 16,
        "t_input": 10.0,
        "t_output": 3.0,
        "n_input": 100,
        "n_output": 10,
    }
    defaults.update(overrides)
    return DataGenerationCapabilities(**defaults)


def _health(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    *,
    data_generation: Mapping[str, Any] | None | str = "default",
) -> HealthMetadata:
    """Build a HealthMetadata from the shared fixture with optional override."""
    payload = json_fixture_loader("health_metadata")
    if data_generation == "default":
        return HealthMetadata.from_payload(payload)
    payload["data_generation"] = (
        dict(data_generation)
        if isinstance(data_generation, Mapping)
        else data_generation
    )
    return HealthMetadata.from_payload(payload)


def test_health_metadata_parses_data_generation_block(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Health metadata parses data generation block."""
    health = HealthMetadata.from_payload(json_fixture_loader("health_metadata"))

    assert health.max_sample_count == 4096
    assert health.decoding_strategy == "parallel_dense"
    assert health.data_generation == _capabilities()


def test_health_metadata_accepts_null_data_generation(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Health metadata accepts null data generation."""
    payload = json_fixture_loader("health_metadata")
    payload["data_generation"] = None

    health = HealthMetadata.from_payload(payload)

    assert health.data_generation is None


def test_plan_forecast_columns_preserves_features_when_supported(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns preserves features when supported."""
    health = _health(json_fixture_loader)

    plan = plan_forecast_columns(
        health=health,
        feature_columns=["equity_index_level", "treasury_10y_yield", "eur_usd_rate"],
        target_columns=["portfolio_nav", "realized_volatility"],
        history_length=100,
        query_times_length=10,
    )

    assert isinstance(plan, ForecastPlan)
    assert plan.feature_columns == (
        "equity_index_level",
        "treasury_10y_yield",
        "eur_usd_rate",
    )
    assert plan.target_columns == ("portfolio_nav", "realized_volatility")
    assert plan.requested_columns == ("portfolio_nav", "realized_volatility")
    assert plan.columns == (
        ColumnSpec(name="equity_index_level", modality="numeric", role="feature"),
        ColumnSpec(name="treasury_10y_yield", modality="numeric", role="feature"),
        ColumnSpec(name="eur_usd_rate", modality="numeric", role="feature"),
        ColumnSpec(name="portfolio_nav", modality="numeric", role="target"),
        ColumnSpec(name="realized_volatility", modality="numeric", role="target"),
    )


def test_plan_forecast_columns_spends_one_budget_across_both_roles(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A split that fits only when both roles share one budget is admitted."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_series": 1,
            "max_series": 5,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )

    plan = plan_forecast_columns(
        health=health,
        feature_columns=["equity_index_level", "treasury_10y_yield", "eur_usd_rate"],
        target_columns=["portfolio_nav", "realized_volatility"],
        history_length=100,
        query_times_length=10,
    )

    # Both roles survive as declared: the envelope budgets width, not roles.
    assert plan.feature_columns == (
        "equity_index_level",
        "treasury_10y_yield",
        "eur_usd_rate",
    )
    assert plan.target_columns == ("portfolio_nav", "realized_volatility")
    assert len(plan.columns) == 5


def test_plan_forecast_columns_raises_when_combined_width_exceeds_max_series(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Features and targets are counted together against ``max_series``."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_series": 1,
            "max_series": 3,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )

    # Neither role alone exceeds 3; their sum does. Under the old per-role caps
    # this request was admissible, which is exactly the regression to guard.
    with pytest.raises(JointFMCapacityError, match="max_series=3"):
        plan_forecast_columns(
            health=health,
            feature_columns=["a", "b"],
            target_columns=["c", "d"],
            history_length=100,
            query_times_length=10,
        )


def test_plan_forecast_columns_raises_when_history_exceeds_training_window(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns raises when history exceeds training window."""
    health = _health(json_fixture_loader)

    with pytest.raises(JointFMCapacityError, match="n_input=100"):
        plan_forecast_columns(
            health=health,
            feature_columns=[],
            target_columns=["portfolio_nav"],
            history_length=101,
            query_times_length=10,
        )


def test_plan_forecast_columns_accepts_history_shorter_than_training_window(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns accepts history shorter than training window."""
    health = _health(json_fixture_loader)

    plan = plan_forecast_columns(
        health=health,
        feature_columns=[],
        target_columns=["portfolio_nav"],
        history_length=50,
        query_times_length=5,
    )

    assert plan.target_columns == ("portfolio_nav",)


def test_plan_forecast_columns_raises_when_query_times_exceed_horizon(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns raises when query times exceed horizon."""
    health = _health(json_fixture_loader)

    with pytest.raises(JointFMCapacityError, match="n_output=10"):
        plan_forecast_columns(
            health=health,
            feature_columns=[],
            target_columns=["portfolio_nav"],
            history_length=100,
            query_times_length=11,
        )


def test_plan_forecast_columns_raises_when_data_generation_missing(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns raises when data generation missing."""
    health = _health(json_fixture_loader, data_generation=None)

    with pytest.raises(JointFMCapacityError, match="data_generation"):
        plan_forecast_columns(
            health=health,
            feature_columns=[],
            target_columns=["portfolio_nav"],
            history_length=100,
            query_times_length=10,
        )


def test_plan_forecast_columns_enforces_min_series(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """A request narrower than the deployment's minimum width is rejected."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_series": 4,
            "max_series": 12,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )

    with pytest.raises(JointFMCapacityError, match="min_series=4"):
        plan_forecast_columns(
            health=health,
            feature_columns=["a", "b"],
            target_columns=["c"],
            history_length=100,
            query_times_length=10,
        )


def test_plan_forecast_columns_rejects_duplicate_column_names(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns rejects duplicate column names."""
    health = _health(json_fixture_loader)

    with pytest.raises(JointFMCapacityError, match="duplicates"):
        plan_forecast_columns(
            health=health,
            feature_columns=["shared"],
            target_columns=["shared"],
            history_length=100,
            query_times_length=10,
        )


def test_plan_forecast_columns_requires_at_least_one_target(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns requires at least one target."""
    health = _health(json_fixture_loader)

    with pytest.raises(JointFMCapacityError, match="at least one target"):
        plan_forecast_columns(
            health=health,
            feature_columns=["a"],
            target_columns=[],
            history_length=100,
            query_times_length=10,
        )
