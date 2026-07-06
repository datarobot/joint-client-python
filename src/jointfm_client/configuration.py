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

"""Structured configuration models and YAML loading for the JointFM SDK."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml
from yaml import YAMLError

from jointfm_client.contract import DEFAULT_CALENDAR_ID, SCHEMA_VERSION
from jointfm_client.exceptions import JointFMConfigurationError


DEFAULT_CONFIG_PATH: Final = Path("config.yaml")
DEFAULT_CONFIG_SAMPLE_PATH: Final = Path("config.sample.yaml")
DEFAULT_DOTENV_PATH: Final = Path(".env")


class _ConfigModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class PathConfig(_ConfigModel):
    """Default local file names used by SDK configuration loading."""

    config: str = str(DEFAULT_CONFIG_PATH)
    sample: str = str(DEFAULT_CONFIG_SAMPLE_PATH)
    dotenv: str = str(DEFAULT_DOTENV_PATH)


class EnvironmentVariableConfig(_ConfigModel):
    """Environment variable names consumed by SDK settings."""

    datarobot_endpoint: str = "DATAROBOT_ENDPOINT"
    datarobot_api_token: str = "DATAROBOT_API_TOKEN"
    deployment_id: str = "JOINTFM_DEPLOYMENT_ID"
    deployment_url: str = "JOINTFM_DEPLOYMENT_URL"
    predict_url: str = "JOINTFM_PREDICT_URL"
    deployment_target: str = "JOINTFM_DEPLOYMENT_TARGET"
    local_base_url: str = "JOINTFM_LOCAL_BASE_URL"
    pulumi_outputs_path: str = "JOINTFM_PULUMI_OUTPUTS_PATH"
    schema_version: str = "JOINTFM_SCHEMA_VERSION"
    model_version: str = "JOINTFM_MODEL_VERSION"

    @field_validator("*")
    @classmethod
    def _validate_environment_name(cls, value: str) -> str:
        if (
            value == ""
            or value.strip() != value
            or any(character.isspace() for character in value)
        ):
            raise ValueError(
                "environment variable names must be non-empty and whitespace-free"
            )
        return value

    def deployment_selector_names(self) -> tuple[str, str, str, str, str]:
        """Return the environment names that select one service target."""
        return (
            self.deployment_id,
            self.deployment_url,
            self.predict_url,
            self.deployment_target,
            self.local_base_url,
        )


class HostedDeploymentConfig(_ConfigModel):
    """Optional service target values layered below .env and process env."""

    datarobot_endpoint: str | None = None
    datarobot_api_token: str | None = Field(default=None, repr=False)
    deployment_id: str | None = None
    deployment_url: str | None = None
    predict_url: str | None = None
    deployment_target: str | None = None
    local_base_url: str | None = None
    pulumi_outputs_path: str | None = None
    schema_version: str | None = None
    model_version: str | None = None

    def to_environment_values(
        self,
        environment: EnvironmentVariableConfig,
    ) -> dict[str, str]:
        """Return configured deployment values keyed by their environment names."""
        values: dict[str, str] = {}
        _set_if_configured(
            values, environment.datarobot_endpoint, self.datarobot_endpoint
        )
        _set_if_configured(
            values, environment.datarobot_api_token, self.datarobot_api_token
        )
        _set_if_configured(values, environment.deployment_id, self.deployment_id)
        _set_if_configured(values, environment.deployment_url, self.deployment_url)
        _set_if_configured(values, environment.predict_url, self.predict_url)
        _set_if_configured(
            values, environment.deployment_target, self.deployment_target
        )
        _set_if_configured(values, environment.local_base_url, self.local_base_url)
        _set_if_configured(
            values, environment.pulumi_outputs_path, self.pulumi_outputs_path
        )
        _set_if_configured(values, environment.schema_version, self.schema_version)
        _set_if_configured(values, environment.model_version, self.model_version)
        return values


class TimeoutConfig(_ConfigModel):
    """Default connect and read timeout values for HTTP requests."""

    connect_seconds: float = 5.0
    read_seconds: float = 60.0

    @field_validator("connect_seconds", "read_seconds")
    @classmethod
    def _validate_positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("timeout values must be finite and positive")
        return value


class RetryConfig(_ConfigModel):
    """Default retry policy for transient HTTP failures."""

    max_attempts: int = 3
    backoff_seconds: float = 1
    max_backoff_seconds: float = 30.0
    status_codes: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

    @field_validator("max_attempts")
    @classmethod
    def _validate_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_attempts must be at least 1")
        return value

    @field_validator("backoff_seconds")
    @classmethod
    def _validate_backoff_seconds(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("backoff_seconds must be finite and non-negative")
        return value

    @field_validator("max_backoff_seconds")
    @classmethod
    def _validate_max_backoff_seconds(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("max_backoff_seconds must be finite and positive")
        return value

    @field_validator("status_codes")
    @classmethod
    def _validate_status_codes(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        for status_code in values:
            if status_code < 400:
                raise ValueError("retry status_codes must be HTTP error statuses")
        return values


class TransportConfig(_ConfigModel):
    """Default HTTP transport settings."""

    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    response_body_excerpt_characters: int = 1024
    retryable_methods: tuple[str, ...] = ("GET", "POST")
    datarobot_request_id_headers: tuple[str, ...] = (
        "X-DataRobot-Request-ID",
        "X-Request-ID",
        "X-DataRobot-Execution-ID",
    )
    user_agent_header: str = "User-Agent"

    @field_validator("response_body_excerpt_characters")
    @classmethod
    def _validate_excerpt_length(cls, value: int) -> int:
        if value < 1:
            raise ValueError("response_body_excerpt_characters must be positive")
        return value

    @field_validator("retryable_methods", "datarobot_request_id_headers")
    @classmethod
    def _validate_non_empty_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("configuration sequences must not be empty")
        for value in values:
            if value == "" or value.strip() != value:
                raise ValueError("configuration sequence entries must be non-empty")
        return values

    @field_validator("user_agent_header")
    @classmethod
    def _validate_user_agent_header(cls, value: str) -> str:
        if value == "" or value.strip() != value:
            raise ValueError("user_agent_header must be non-empty")
        return value


class ForecastConfig(_ConfigModel):
    """Default forecast request values shared by client and adapter helpers."""

    schema_version: str = SCHEMA_VERSION
    query_mode: str = "forecast"
    return_mode: str = "mean"
    time_index_mode: str = "ordinal"
    use_local_normalized_time: bool = False
    calendar_id: str = DEFAULT_CALENDAR_ID
    ordinal_step: int = 1

    @field_validator(
        "schema_version", "query_mode", "return_mode", "time_index_mode", "calendar_id"
    )
    @classmethod
    def _validate_plain_string(cls, value: str) -> str:
        if value == "" or value.strip() != value:
            raise ValueError("forecast string defaults must be non-empty")
        return value

    @field_validator("ordinal_step")
    @classmethod
    def _validate_ordinal_step(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ordinal_step must be positive")
        return value


class ForecastCsvConfig(_ConfigModel):
    """Default values used by the forecast-csv CLI command."""

    time_index_mode: str = "ordinal"
    return_mode: str = "mean"


class CLIConfig(_ConfigModel):
    """Default command line interface settings."""

    dotenv_path: str = str(DEFAULT_DOTENV_PATH)
    forecast_csv: ForecastCsvConfig = Field(default_factory=ForecastCsvConfig)


class JointFMConfig(_ConfigModel):
    """Top-level SDK configuration loaded from defaults and optional YAML."""

    paths: PathConfig = Field(default_factory=PathConfig)
    environment: EnvironmentVariableConfig = Field(
        default_factory=EnvironmentVariableConfig
    )
    deployment: HostedDeploymentConfig = Field(default_factory=HostedDeploymentConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)


def load_configuration(
    *,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    overrides: JointFMConfig | Mapping[str, Any] | None = None,
) -> JointFMConfig:
    """Load structured SDK configuration from defaults, YAML, and overrides."""
    payload = _read_yaml_mapping(config_path)
    override_payload = _override_payload(overrides)
    if override_payload:
        payload = _deep_merge(payload, override_payload)
    try:
        return JointFMConfig.model_validate(payload)
    except ValidationError as error:
        raise JointFMConfigurationError(
            f"Invalid JointFM configuration: {error}"
        ) from error


def _read_yaml_mapping(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise JointFMConfigurationError(
            f"Unable to read configuration file {path}"
        ) from error
    except YAMLError as error:
        raise JointFMConfigurationError(
            f"Configuration file {path} must contain valid YAML"
        ) from error

    if raw_payload is None:
        return {}
    if not isinstance(raw_payload, Mapping):
        raise JointFMConfigurationError(
            f"Configuration file {path} must contain a YAML mapping"
        )
    return dict(raw_payload)


def _override_payload(
    overrides: JointFMConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, JointFMConfig):
        return overrides.model_dump(mode="python")
    if not isinstance(overrides, Mapping):
        raise JointFMConfigurationError("configuration overrides must be a mapping")
    return dict(overrides)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(override_value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], override_value)
            continue
        merged[key] = override_value
    return merged


def _set_if_configured(values: dict[str, str], name: str, value: str | None) -> None:
    if value is not None:
        values[name] = value


DEFAULT_ENVIRONMENT_CONFIG: Final = EnvironmentVariableConfig()
DATAROBOT_ENDPOINT_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.datarobot_endpoint
DATAROBOT_API_TOKEN_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.datarobot_api_token
JOINTFM_DEPLOYMENT_ID_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.deployment_id
JOINTFM_DEPLOYMENT_URL_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.deployment_url
JOINTFM_PREDICT_URL_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.predict_url
JOINTFM_DEPLOYMENT_TARGET_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.deployment_target
JOINTFM_LOCAL_BASE_URL_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.local_base_url
JOINTFM_PULUMI_OUTPUTS_PATH_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.pulumi_outputs_path
JOINTFM_SCHEMA_VERSION_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.schema_version
JOINTFM_MODEL_VERSION_ENV: Final = DEFAULT_ENVIRONMENT_CONFIG.model_version

DEFAULT_TRANSPORT_CONFIG: Final = TransportConfig()
DEFAULT_TIMEOUT_CONFIG: Final = DEFAULT_TRANSPORT_CONFIG.timeout
DEFAULT_RETRY_CONFIG: Final = DEFAULT_TRANSPORT_CONFIG.retry
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final = DEFAULT_TIMEOUT_CONFIG.connect_seconds
DEFAULT_READ_TIMEOUT_SECONDS: Final = DEFAULT_TIMEOUT_CONFIG.read_seconds
DEFAULT_MAX_ATTEMPTS: Final = DEFAULT_RETRY_CONFIG.max_attempts
DEFAULT_BACKOFF_SECONDS: Final = DEFAULT_RETRY_CONFIG.backoff_seconds
DEFAULT_MAX_BACKOFF_SECONDS: Final = DEFAULT_RETRY_CONFIG.max_backoff_seconds
DEFAULT_RETRY_STATUS_CODES: Final = DEFAULT_RETRY_CONFIG.status_codes
DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS: Final = (
    DEFAULT_TRANSPORT_CONFIG.response_body_excerpt_characters
)
DEFAULT_RETRYABLE_METHODS: Final = DEFAULT_TRANSPORT_CONFIG.retryable_methods
DATAROBOT_REQUEST_ID_HEADERS: Final = (
    DEFAULT_TRANSPORT_CONFIG.datarobot_request_id_headers
)
USER_AGENT_HEADER: Final = DEFAULT_TRANSPORT_CONFIG.user_agent_header

DEFAULT_FORECAST_CONFIG: Final = ForecastConfig()
DEFAULT_FORECAST_SCHEMA_VERSION: Final = DEFAULT_FORECAST_CONFIG.schema_version
DEFAULT_QUERY_MODE: Final = DEFAULT_FORECAST_CONFIG.query_mode
DEFAULT_RETURN_MODE: Final = DEFAULT_FORECAST_CONFIG.return_mode
DEFAULT_TIME_INDEX_MODE: Final = DEFAULT_FORECAST_CONFIG.time_index_mode
DEFAULT_USE_LOCAL_NORMALIZED_TIME: Final = (
    DEFAULT_FORECAST_CONFIG.use_local_normalized_time
)
DEFAULT_ORDINAL_STEP: Final = DEFAULT_FORECAST_CONFIG.ordinal_step

DEFAULT_CLI_CONFIG: Final = CLIConfig()
DEFAULT_CLI_DOTENV_PATH: Final = Path(DEFAULT_CLI_CONFIG.dotenv_path)
DEFAULT_CLI_TIME_INDEX_MODE: Final = DEFAULT_CLI_CONFIG.forecast_csv.time_index_mode
DEFAULT_CLI_RETURN_MODE: Final = DEFAULT_CLI_CONFIG.forecast_csv.return_mode


__all__ = [
    "CLIConfig",
    "DATAROBOT_API_TOKEN_ENV",
    "DATAROBOT_ENDPOINT_ENV",
    "DATAROBOT_REQUEST_ID_HEADERS",
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_CLI_CONFIG",
    "DEFAULT_CLI_DOTENV_PATH",
    "DEFAULT_CLI_RETURN_MODE",
    "DEFAULT_CLI_TIME_INDEX_MODE",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CONFIG_SAMPLE_PATH",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_DOTENV_PATH",
    "DEFAULT_ENVIRONMENT_CONFIG",
    "DEFAULT_FORECAST_CONFIG",
    "DEFAULT_FORECAST_SCHEMA_VERSION",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "DEFAULT_ORDINAL_STEP",
    "DEFAULT_QUERY_MODE",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS",
    "DEFAULT_RETRYABLE_METHODS",
    "DEFAULT_RETRY_CONFIG",
    "DEFAULT_RETRY_STATUS_CODES",
    "DEFAULT_RETURN_MODE",
    "DEFAULT_TIME_INDEX_MODE",
    "DEFAULT_TIMEOUT_CONFIG",
    "DEFAULT_TRANSPORT_CONFIG",
    "DEFAULT_USE_LOCAL_NORMALIZED_TIME",
    "EnvironmentVariableConfig",
    "ForecastConfig",
    "ForecastCsvConfig",
    "HostedDeploymentConfig",
    "JOINTFM_DEPLOYMENT_ID_ENV",
    "JOINTFM_DEPLOYMENT_TARGET_ENV",
    "JOINTFM_DEPLOYMENT_URL_ENV",
    "JOINTFM_LOCAL_BASE_URL_ENV",
    "JOINTFM_MODEL_VERSION_ENV",
    "JOINTFM_PREDICT_URL_ENV",
    "JOINTFM_PULUMI_OUTPUTS_PATH_ENV",
    "JOINTFM_SCHEMA_VERSION_ENV",
    "JointFMConfig",
    "PathConfig",
    "RetryConfig",
    "TimeoutConfig",
    "TransportConfig",
    "USER_AGENT_HEADER",
    "load_configuration",
]
