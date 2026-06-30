# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixture compatibility surface of jointfm_client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jointfm_client import (
    ForecastResponse,
    HealthMetadata,
    JointFMServiceError,
    MeanForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
    validate_service_metadata,
)


def test_health_fixture_matches_current_v1_service_contract(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    """Health fixture matches current v1 service contract."""
    payload = json_fixture_loader("health_metadata")
    expected_model_version = payload["model_version"]
    assert isinstance(expected_model_version, str)

    validate_service_metadata(
        payload,
        expected_model_version=expected_model_version,
    )
    metadata = HealthMetadata.from_payload(payload)

    assert metadata.schema_version == "v1"
    assert metadata.supported_return_modes == (
        "mean",
        "samples",
        "quantiles",
        "log_prob",
    )


@pytest.mark.parametrize(
    ("request_fixture", "response_fixture", "result_type", "return_mode"),
    [
        (
            "forecast_mean_request",
            "forecast_mean_response",
            MeanForecastResult,
            "mean",
        ),
        (
            "forecast_samples_request",
            "forecast_samples_response",
            SampleForecastResult,
            "samples",
        ),
        (
            "forecast_quantiles_request",
            "forecast_quantiles_response",
            QuantileForecastResult,
            "quantiles",
        ),
    ],
)
def test_checked_in_fixture_payloads_parse_as_forecast_results(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    request_fixture: str,
    response_fixture: str,
    result_type: type[ForecastResponse],
    return_mode: str,
) -> None:
    """Checked in fixture payloads parse as forecast results."""
    request_payload = json_fixture_loader(request_fixture)
    response_payload = json_fixture_loader(response_fixture)

    result = ForecastResponse.from_payload(
        response_payload,
        request_payload=request_payload,
    )

    assert isinstance(result, result_type)
    assert result.return_mode == return_mode


@pytest.mark.parametrize(
    ("fixture_name", "error_code", "message_fragment"),
    [
        (
            "validation_error_response",
            "VALIDATION_ERROR",
            "query_times must not be empty",
        ),
        (
            "schema_version_mismatch_response",
            "SCHEMA_VERSION_MISMATCH",
            "Unsupported schema_version",
        ),
        (
            "model_version_mismatch_response",
            "MODEL_VERSION_MISMATCH",
            "Unsupported model_version",
        ),
        (
            "input_size_exceeded_response",
            "INPUT_SIZE_EXCEEDED",
            "n_samples exceeds the configured container cap",
        ),
    ],
)
def test_checked_in_error_fixtures_raise_typed_service_errors(
    json_fixture_loader: Callable[[str], dict[str, Any]],
    fixture_name: str,
    error_code: str,
    message_fragment: str,
) -> None:
    """Checked in error fixtures raise typed service errors."""
    with pytest.raises(JointFMServiceError, match=message_fragment) as exc_info:
        ForecastResponse.raise_for_errors(json_fixture_loader(fixture_name))

    assert exc_info.value.jointfm_errors[0]["code"] == error_code
