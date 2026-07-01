# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capabilities surface of jointfm_client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
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
        "min_features": 0,
        "max_features": 12,
        "min_targets": 1,
        "max_targets": 4,
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


def test_plan_forecast_columns_downgrades_features_when_max_features_is_zero(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Plan forecast columns downgrades features when max features is zero."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_features": 0,
            "max_features": 0,
            "min_targets": 1,
            "max_targets": 5,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )
    caplog.set_level(logging.WARNING, logger="jointfm_client.capabilities")

    plan = plan_forecast_columns(
        health=health,
        feature_columns=["equity_index_level", "treasury_10y_yield", "eur_usd_rate"],
        target_columns=["portfolio_nav", "realized_volatility"],
        history_length=100,
        query_times_length=10,
    )

    assert plan.feature_columns == ()
    assert plan.target_columns == (
        "equity_index_level",
        "treasury_10y_yield",
        "eur_usd_rate",
        "portfolio_nav",
        "realized_volatility",
    )
    assert plan.requested_columns == ("portfolio_nav", "realized_volatility")
    assert all(column.role == "target" for column in plan.columns)
    assert any("max_features=0" in record.message for record in caplog.records)


def test_plan_forecast_columns_raises_when_targets_exceed_capacity_after_downgrade(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns raises when targets exceed capacity after downgrade."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_features": 0,
            "max_features": 0,
            "min_targets": 1,
            "max_targets": 3,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )

    with pytest.raises(JointFMCapacityError, match="max_targets=3"):
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


def test_plan_forecast_columns_enforces_min_targets(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Plan forecast columns enforces min targets."""
    health = _health(
        json_fixture_loader,
        data_generation={
            "sampler_type": "studentt",
            "min_features": 0,
            "max_features": 12,
            "min_targets": 2,
            "max_targets": 4,
            "t_input": 10.0,
            "t_output": 3.0,
            "n_input": 100,
            "n_output": 10,
        },
    )

    with pytest.raises(JointFMCapacityError, match="min_targets=2"):
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
