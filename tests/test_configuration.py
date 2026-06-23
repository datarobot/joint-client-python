from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

import jointfm_client.client as client_module
from jointfm_client import (
    DATAROBOT_API_TOKEN_ENV,
    DATAROBOT_ENDPOINT_ENV,
    JOINTFM_DEPLOYMENT_ID_ENV,
    JOINTFM_MODEL_VERSION_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
    JointFMClient,
    JointFMConfig,
    JointFMConfigurationError,
    JointFMRetryConfig,
    JointFMSettings,
    JointFMTimeoutConfig,
    load_configuration,
    load_settings,
)


class _HealthTransport:
    _METADATA: dict[str, object] = {
        "status": "ok",
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.yaml",
        "checkpoint_version": "yaml",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "studentt",
        "supported_query_modes": ["forecast"],
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "default_sample_count": 256,
        "max_sample_count": 4096,
    }

    def get_json(self, url: str) -> dict[str, object]:
        del url
        return dict(self._METADATA)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, object]:
        del url
        if payload.get("request_type") == "health":
            return dict(self._METADATA)
        return {}


def test_checked_in_yaml_sample_matches_configuration_defaults() -> None:
    repo_root = Path(__file__).parents[1]
    expected_defaults = JointFMConfig().model_dump(mode="json")
    config_path = repo_root / "config.sample.yaml"

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == expected_defaults
    assert load_configuration(config_path=config_path).model_dump(mode="json") == expected_defaults


def test_load_configuration_layers_yaml_over_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "deployment:",
                "  datarobot_endpoint: https://app.datarobot.com/api/v2",
                "  datarobot_api_token: yaml-token",
                "transport:",
                "  timeout:",
                "    connect_seconds: 1.25",
                "  retry:",
                "    max_attempts: 2",
                "    status_codes: [500]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_configuration(config_path=config_path)

    assert config.deployment.datarobot_endpoint == "https://app.datarobot.com/api/v2"
    assert config.deployment.datarobot_api_token == "yaml-token"
    assert config.transport.timeout.connect_seconds == 1.25
    assert config.transport.timeout.read_seconds == 60.0
    assert config.transport.retry.max_attempts == 2
    assert config.transport.retry.backoff_seconds == 0.25
    assert config.transport.retry.status_codes == (500,)


def test_load_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("transport:\n  unknown: true\n", encoding="utf-8")

    with pytest.raises(JointFMConfigurationError, match="Invalid JointFM configuration"):
        load_configuration(config_path=config_path)


def test_load_settings_layers_config_below_dotenv_and_environment(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "datarobot_endpoint": "https://app.datarobot.com/api/v2",
                    "datarobot_api_token": "yaml-token",
                    "deployment_id": "yaml-deployment-id",
                    "schema_version": "v1",
                    "model_version": "jointfm-inference:0.2.0+ckpt.yaml",
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "DATAROBOT_API_TOKEN=dotenv-token",
                "JOINTFM_DEPLOYMENT_ID=dotenv-deployment-id",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        env={JOINTFM_DEPLOYMENT_ID_ENV: "env-deployment-id"},
        dotenv_path=dotenv_path,
        config_path=config_path,
    )

    assert settings.datarobot_endpoint == "https://app.datarobot.com/api/v2"
    assert settings.datarobot_api_token == "dotenv-token"
    assert settings.deployment_id == "env-deployment-id"
    assert settings.model_version == "jointfm-inference:0.2.0+ckpt.yaml"


def test_client_from_env_uses_transport_defaults_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "datarobot_endpoint": "https://app.datarobot.com/api/v2",
                    "datarobot_api_token": "yaml-token",
                    "deployment_id": "yaml-deployment-id",
                    "schema_version": "v1",
                    "model_version": "jointfm-inference:0.2.0+ckpt.yaml",
                },
                "transport": {
                    "timeout": {"connect_seconds": 1.0, "read_seconds": 2.0},
                    "retry": {
                        "max_attempts": 2,
                        "backoff_seconds": 0.1,
                        "status_codes": [500],
                    },
                    "retryable_methods": ["GET"],
                    "response_body_excerpt_characters": 64,
                    "datarobot_request_id_headers": ["X-Custom-Request-ID"],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    captured_timeout: JointFMTimeoutConfig | None = None
    captured_retry_config: JointFMRetryConfig | None = None
    captured_response_body_excerpt_characters: int | None = None
    captured_datarobot_request_id_headers: tuple[str, ...] | None = None

    def capture_transport(
        settings: JointFMSettings,
        *,
        session: requests.Session | None = None,
        timeout: JointFMTimeoutConfig = JointFMTimeoutConfig(),
        retry_config: JointFMRetryConfig = JointFMRetryConfig(),
        user_agent: str | None = None,
        response_body_excerpt_characters: int = 1024,
        datarobot_request_id_headers: tuple[str, ...] = (
            "X-DataRobot-Request-ID",
            "X-Request-ID",
            "X-DataRobot-Execution-ID",
        ),
    ) -> _HealthTransport:
        del settings, session, user_agent
        nonlocal captured_timeout, captured_retry_config
        nonlocal captured_response_body_excerpt_characters
        nonlocal captured_datarobot_request_id_headers
        captured_timeout = timeout
        captured_retry_config = retry_config
        captured_response_body_excerpt_characters = response_body_excerpt_characters
        captured_datarobot_request_id_headers = datarobot_request_id_headers
        return _HealthTransport()

    monkeypatch.setattr(
        client_module.JointFMHTTPTransport,
        "from_settings",
        capture_transport,
    )

    client = JointFMClient.from_env(env={}, dotenv_path=None, config_path=config_path)

    assert client.health().model_version == "jointfm-inference:0.2.0+ckpt.yaml"
    assert captured_timeout == JointFMTimeoutConfig(connect_seconds=1.0, read_seconds=2.0)
    assert captured_retry_config == JointFMRetryConfig(
        max_attempts=2,
        backoff_seconds=0.1,
        status_codes=(500,),
        allowed_methods=("GET",),
    )
    assert captured_response_body_excerpt_characters == 64
    assert captured_datarobot_request_id_headers == ("X-Custom-Request-ID",)


def test_configuration_override_mapping_wins_over_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({"transport": {"retry": {"max_attempts": 2}}}),
        encoding="utf-8",
    )

    config = load_configuration(
        config_path=config_path,
        overrides={"transport": {"retry": {"max_attempts": 4}}},
    )

    assert config.transport.retry.max_attempts == 4