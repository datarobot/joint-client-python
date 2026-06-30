# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cli surface of jointfm_client."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd

from jointfm_client import HealthMetadata, JointFMResponseDecodeError, JointFMSettings
from jointfm_client import cli


class FakeHealthClient:
    """Fake Health Client (test helper)."""

    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url=(
            "https://app.datarobot.com/api/v2/deployments/"
            "deployment-id/predictionsUnstructured"
        ),
        predict_url=(
            "https://app.datarobot.com/api/v2/deployments/"
            "deployment-id/predictionsUnstructured"
        ),
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )

    def health(self) -> HealthMetadata:
        """Health."""
        return HealthMetadata(
            status="ok",
            schema_version="v1",
            image_version="0.2.0",
            model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
            checkpoint_version="sdk-test",
            checkpoint_path="/models/jointfm.pt",
            device="cpu",
            head="studentt",
            supported_query_modes=("forecast",),
            supported_return_modes=("mean", "samples", "quantiles", "log_prob"),
            supported_time_index_modes=(
                "ordinal",
                "continuous_float",
                "absolute_datetime",
            ),
            time_index_encoding="legacy_discrete_grid",
            default_sample_count=256,
            max_sample_count=4096,
        )


class FakePredictClient:
    """Fake Predict Client (test helper)."""

    def __init__(self) -> None:
        """Init."""
        self.payload: Mapping[str, Any] | None = None

    def predict(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Predict."""
        self.payload = payload
        return {"ok": True, "model_version": payload["model_version"]}


class FakeForecastResult:
    """Fake Forecast Result (test helper)."""

    def to_pandas_tidy(self) -> pd.DataFrame:
        """To pandas tidy."""
        return pd.DataFrame.from_records(
            [
                {
                    "query_time": 2,
                    "requested_column": "target",
                    "value": 12.0,
                }
            ]
        )


class FakeForecastClient:
    """Fake Forecast Client (test helper)."""

    def __init__(self) -> None:
        """Init."""
        self.kwargs: dict[str, Any] | None = None

    def forecast(self, history: Any, **kwargs: Any) -> FakeForecastResult:
        """Forecast."""
        self.kwargs = kwargs
        assert list(history.columns) == ["target"]
        return FakeForecastResult()


def test_health_command_prints_non_secret_metadata(monkeypatch, capsys) -> None:
    """Health command prints non secret metadata."""
    monkeypatch.setattr(
        cli.JointFMClient, "from_env", lambda *, dotenv_path: FakeHealthClient()
    )

    exit_code = cli.main(["health", "--no-dotenv"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["service"]["status"] == "ok"
    assert payload["deployment"]["deployment_id"] == "deployment-id"
    assert "secret-token" not in output


def test_predict_command_writes_response_file(monkeypatch, tmp_path: Path) -> None:
    """Predict command writes response file."""
    client = FakePredictClient()
    monkeypatch.setattr(cli.JointFMClient, "from_env", lambda *, dotenv_path: client)
    request_file = tmp_path / "request.json"
    response_file = tmp_path / "response.json"
    request_file.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "model_version": "jointfm-inference:0.2.0+ckpt.sdk-test",
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        ["predict", "--no-dotenv", str(request_file), str(response_file)]
    )

    assert exit_code == 0
    assert client.payload is not None
    assert json.loads(response_file.read_text(encoding="utf-8")) == {
        "ok": True,
        "model_version": "jointfm-inference:0.2.0+ckpt.sdk-test",
    }


def test_predict_command_rejects_non_object_request(tmp_path: Path, capsys) -> None:
    """Predict command rejects non object request."""
    request_file = tmp_path / "request.json"
    response_file = tmp_path / "response.json"
    request_file.write_text("[]", encoding="utf-8")

    exit_code = cli.main(
        ["predict", "--no-dotenv", str(request_file), str(response_file)]
    )

    assert exit_code == 2
    assert "must contain a JSON object" in capsys.readouterr().err
    assert not response_file.exists()


def test_command_errors_include_response_metadata(monkeypatch, capsys) -> None:
    """Command errors include response metadata."""

    def raise_decode_error(*, dotenv_path: Path | None) -> None:
        """Raise decode error."""
        del dotenv_path
        raise JointFMResponseDecodeError(
            "JointFM service returned a non-JSON response body",
            status_code=404,
            response_body_excerpt="<html>not found</html>",
            datarobot_request_id="request-id",
        )

    monkeypatch.setattr(cli.JointFMClient, "from_env", raise_decode_error)

    exit_code = cli.main(["health", "--no-dotenv"])

    error_output = capsys.readouterr().err
    assert exit_code == 2
    assert "non-JSON response body" in error_output
    assert "HTTP 404" in error_output
    assert "request-id" in error_output
    assert "<html>not found</html>" in error_output


def test_forecast_csv_command_writes_tidy_output(monkeypatch, tmp_path: Path) -> None:
    """Forecast csv command writes tidy output."""
    client = FakeForecastClient()
    monkeypatch.setattr(cli.JointFMClient, "from_env", lambda *, dotenv_path: client)
    history_file = tmp_path / "history.csv"
    output_file = tmp_path / "forecast.csv"
    history_file.write_text("target\n10.0\n11.0\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "forecast-csv",
            "--no-dotenv",
            str(history_file),
            str(output_file),
            "--query-times",
            "2",
            "--target-column",
            "target",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert client.kwargs is not None
    assert client.kwargs["query_times"] == [2]
    assert client.kwargs["requested_columns"] == ["target"]
    assert client.kwargs["target_columns"] == ["target"]
    assert client.kwargs["seed"] == 7
    assert output_file.read_text(encoding="utf-8") == (
        "query_time,requested_column,value\n2,target,12.0\n"
    )
