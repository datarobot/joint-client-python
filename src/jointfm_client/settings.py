"""Settings and URL helpers for DataRobot-hosted JointFM calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import urljoin, urlparse

from dotenv import dotenv_values

from jointfm_client.configuration import (
    DATAROBOT_API_TOKEN_ENV,
    DATAROBOT_ENDPOINT_ENV,
    DEFAULT_CONFIG_PATH,
    EnvironmentVariableConfig,
    JOINTFM_DEPLOYMENT_ID_ENV,
    JOINTFM_DEPLOYMENT_TARGET_ENV,
    JOINTFM_DEPLOYMENT_URL_ENV,
    JOINTFM_LOCAL_BASE_URL_ENV,
    JOINTFM_MODEL_VERSION_ENV,
    JOINTFM_PREDICT_URL_ENV,
    JOINTFM_PULUMI_OUTPUTS_PATH_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
    JointFMConfig,
    load_configuration,
)
from jointfm_client.contract import (
    DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE,
    LOCAL_HEALTH_ROUTE,
    LOCAL_PREDICT_ROUTE,
    SCHEMA_VERSION,
)
from jointfm_client.exceptions import JointFMConfigurationError

DeploymentSelector: TypeAlias = Literal[
    "deployment_id",
    "deployment_url",
    "predict_url",
    "pulumi_target",
    "local_service",
]


@dataclass(frozen=True, slots=True)
class JointFMSettings:
    """Validated settings for one hosted or local JointFM service target."""

    datarobot_endpoint: str | None
    datarobot_api_token: str | None = field(repr=False)
    health_url: str
    predict_url: str
    deployment_selector: DeploymentSelector
    schema_version: str
    model_version: str | None = None
    deployment_id: str | None = None
    deployment_url: str | None = None
    deployment_target: str | None = None
    local_base_url: str | None = None


def load_settings(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = ".env",
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    config: JointFMConfig | Mapping[str, Any] | None = None,
) -> JointFMSettings:
    """Load and validate hosted or local JointFM settings."""

    jointfm_config = load_configuration(config_path=config_path, overrides=config)
    environment = jointfm_config.environment
    env_values = _merge_config_dotenv_and_environment(
        config=jointfm_config,
        env=env,
        dotenv_path=dotenv_path,
    )
    schema_version = validate_jointfm_schema_version(
        _required_env(env_values, environment.schema_version)
    )
    model_version = _optional_model_version(env_values, environment.model_version)
    selector_name = _resolve_single_deployment_selector(env_values, environment)

    if selector_name == environment.local_base_url:
        local_base_url = normalize_local_service_base_url(
            _required_env(env_values, environment.local_base_url)
        )
        return JointFMSettings(
            datarobot_endpoint=None,
            datarobot_api_token=None,
            health_url=build_local_health_url(local_base_url),
            predict_url=build_local_predict_url(local_base_url),
            deployment_selector="local_service",
            schema_version=schema_version,
            model_version=model_version,
            local_base_url=local_base_url,
        )

    datarobot_endpoint = normalize_datarobot_endpoint(
        _required_env(env_values, environment.datarobot_endpoint)
    )
    datarobot_api_token = validate_datarobot_api_token(
        _required_env(env_values, environment.datarobot_api_token)
    )

    if selector_name == environment.deployment_id:
        deployment_id = normalize_deployment_id(
            _required_env(env_values, environment.deployment_id)
        )
        deployment_url = build_hosted_deployment_url(datarobot_endpoint, deployment_id)
        predict_url = build_hosted_predict_url(datarobot_endpoint, deployment_id)
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="deployment_id",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=deployment_id,
            deployment_url=deployment_url,
        )

    if selector_name == environment.deployment_url:
        deployment_url = normalize_hosted_deployment_url(
            _required_env(env_values, environment.deployment_url)
        )
        deployment_id = deployment_id_from_hosted_deployment_url(deployment_url)
        predict_url = build_hosted_predict_url_from_deployment_url(deployment_url)
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="deployment_url",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=deployment_id,
            deployment_url=deployment_url,
        )

    if selector_name == environment.predict_url:
        predict_url = normalize_hosted_predict_url(
            _required_env(env_values, environment.predict_url)
        )
        deployment_url = deployment_url_from_hosted_predict_url(predict_url)
        deployment_id = deployment_id_from_hosted_deployment_url(deployment_url)
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="predict_url",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=deployment_id,
            deployment_url=deployment_url,
        )

    deployment_target = _required_env(env_values, environment.deployment_target)
    outputs_path = _required_env(env_values, environment.pulumi_outputs_path)
    target_outputs = _load_pulumi_target_outputs(outputs_path, deployment_target)
    deployment_id = _optional_string_output(target_outputs, "deployment_id")
    if deployment_id is not None:
        normalized_deployment_id = normalize_deployment_id(deployment_id)
        deployment_url = build_hosted_deployment_url(
            datarobot_endpoint,
            normalized_deployment_id,
        )
        predict_url = build_hosted_predict_url(
            datarobot_endpoint,
            normalized_deployment_id,
        )
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="pulumi_target",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=normalized_deployment_id,
            deployment_url=deployment_url,
            deployment_target=deployment_target,
        )

    deployment_url_output = _optional_string_output(target_outputs, "deployment_url")
    if deployment_url_output is not None:
        deployment_url = normalize_hosted_deployment_url(deployment_url_output)
        deployment_id = deployment_id_from_hosted_deployment_url(deployment_url)
        predict_url = build_hosted_predict_url_from_deployment_url(deployment_url)
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="pulumi_target",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=deployment_id,
            deployment_url=deployment_url,
            deployment_target=deployment_target,
        )

    predict_url_output = _optional_string_output(target_outputs, "predict_url")
    if predict_url_output is not None:
        predict_url = normalize_hosted_predict_url(predict_url_output)
        deployment_url = deployment_url_from_hosted_predict_url(predict_url)
        deployment_id = deployment_id_from_hosted_deployment_url(deployment_url)
        return JointFMSettings(
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="pulumi_target",
            schema_version=schema_version,
            model_version=model_version,
            deployment_id=deployment_id,
            deployment_url=deployment_url,
            deployment_target=deployment_target,
        )

    raise JointFMConfigurationError(
        "Pulumi output target must contain deployment_id, deployment_url, or predict_url"
    )


def normalize_datarobot_endpoint(value: str) -> str:
    """Return a normalized DataRobot API v2 endpoint URL."""

    normalized_value = _normalize_non_whitespace_string(value, DATAROBOT_ENDPOINT_ENV)
    normalized_endpoint = normalized_value.rstrip("/")
    parsed_endpoint = urlparse(normalized_endpoint)
    if parsed_endpoint.scheme != "https":
        raise JointFMConfigurationError(
            f"{DATAROBOT_ENDPOINT_ENV} must be an https DataRobot API v2 URL"
        )
    if not parsed_endpoint.netloc:
        raise JointFMConfigurationError(
            f"{DATAROBOT_ENDPOINT_ENV} must include a hostname"
        )
    if parsed_endpoint.path != "/api/v2":
        raise JointFMConfigurationError(
            f"{DATAROBOT_ENDPOINT_ENV} must end with /api/v2"
        )
    if parsed_endpoint.params or parsed_endpoint.query or parsed_endpoint.fragment:
        raise JointFMConfigurationError(
            f"{DATAROBOT_ENDPOINT_ENV} must not include params, query, or fragment"
        )
    return normalized_endpoint


def validate_datarobot_api_token(value: str) -> str:
    """Validate a non-empty DataRobot API token without exposing it."""

    return _normalize_non_whitespace_string(value, DATAROBOT_API_TOKEN_ENV)


def validate_jointfm_schema_version(value: str) -> str:
    """Validate the configured JointFM request schema version."""

    schema_version = _normalize_non_whitespace_string(
        value,
        JOINTFM_SCHEMA_VERSION_ENV,
    )
    if schema_version != SCHEMA_VERSION:
        raise JointFMConfigurationError(
            f"{JOINTFM_SCHEMA_VERSION_ENV} must be {SCHEMA_VERSION!r}"
        )
    return schema_version


def validate_jointfm_model_version(value: str) -> str:
    """Validate the configured JointFM deployment model version."""

    return _normalize_non_whitespace_string(value, JOINTFM_MODEL_VERSION_ENV)


def _optional_model_version(
    env: Mapping[str, str],
    name: str,
) -> str | None:
    """Return the configured model version, or None when unset.

    Unset means the SDK will discover the model version from /healthz at
    runtime. When set, the value participates in the existing strict
    drift-check against /healthz as an opt-in safety guard.
    """

    value = env.get(name)
    if value is None or value == "":
        return None
    return validate_jointfm_model_version(value)


def normalize_deployment_id(value: str) -> str:
    """Validate a DataRobot deployment ID as one non-empty path segment."""

    deployment_id = _normalize_non_whitespace_string(value, JOINTFM_DEPLOYMENT_ID_ENV)
    if "/" in deployment_id:
        raise JointFMConfigurationError(
            f"{JOINTFM_DEPLOYMENT_ID_ENV} must be a deployment ID, not a URL"
        )
    return deployment_id


def build_hosted_deployment_url(datarobot_endpoint: str, deployment_id: str) -> str:
    """Build the hosted DataRobot deployment URL for one deployment ID."""

    service_base_url = (
        normalize_datarobot_endpoint(datarobot_endpoint).rstrip("/") + "/"
    )
    normalized_deployment_id = normalize_deployment_id(deployment_id)
    return urljoin(service_base_url, f"deployments/{normalized_deployment_id}")


def build_hosted_predict_url(datarobot_endpoint: str, deployment_id: str) -> str:
    """Build the hosted DataRobot unstructured prediction URL."""

    service_base_url = (
        normalize_datarobot_endpoint(datarobot_endpoint).rstrip("/") + "/"
    )
    normalized_deployment_id = normalize_deployment_id(deployment_id)
    predict_route = DATAROBOT_UNSTRUCTURED_PREDICTION_ROUTE_TEMPLATE.format(
        deployment_id=normalized_deployment_id
    )
    return urljoin(service_base_url, predict_route)


def normalize_hosted_deployment_url(value: str) -> str:
    """Validate a hosted DataRobot deployment URL."""

    deployment_url = _normalize_absolute_https_url(value, JOINTFM_DEPLOYMENT_URL_ENV)
    deployment_id_from_hosted_deployment_url(deployment_url)
    return deployment_url


def normalize_hosted_predict_url(value: str) -> str:
    """Validate a hosted DataRobot unstructured prediction URL."""

    predict_url = _normalize_absolute_https_url(value, JOINTFM_PREDICT_URL_ENV)
    if not predict_url.endswith("/predictionsUnstructured"):
        raise JointFMConfigurationError(
            f"{JOINTFM_PREDICT_URL_ENV} must end with /predictionsUnstructured"
        )
    deployment_url_from_hosted_predict_url(predict_url)
    return predict_url


def build_hosted_predict_url_from_deployment_url(deployment_url: str) -> str:
    """Build the hosted prediction URL from a hosted deployment URL."""

    normalized_deployment_url = normalize_hosted_deployment_url(deployment_url)
    return urljoin(
        normalized_deployment_url.rstrip("/") + "/", "predictionsUnstructured"
    )


def deployment_id_from_hosted_deployment_url(deployment_url: str) -> str:
    """Extract the deployment ID from a hosted deployment URL."""

    parsed_url = urlparse(deployment_url)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if len(path_parts) < 4 or path_parts[-2] != "deployments":
        raise JointFMConfigurationError(
            f"{JOINTFM_DEPLOYMENT_URL_ENV} must end with /deployments/{{deployment_id}}"
        )
    return normalize_deployment_id(path_parts[-1])


def deployment_url_from_hosted_predict_url(predict_url: str) -> str:
    """Return the hosted deployment URL that owns a hosted prediction URL."""

    normalized_predict_url = _normalize_absolute_https_url(
        predict_url,
        JOINTFM_PREDICT_URL_ENV,
    )
    suffix = "/predictionsUnstructured"
    if not normalized_predict_url.endswith(suffix):
        raise JointFMConfigurationError(
            f"{JOINTFM_PREDICT_URL_ENV} must end with {suffix}"
        )
    return normalized_predict_url[: -len(suffix)]


def build_local_health_url(service_base_url: str) -> str:
    """Build the direct local service health URL."""

    return _build_local_service_url(service_base_url, LOCAL_HEALTH_ROUTE)


def build_local_predict_url(service_base_url: str) -> str:
    """Build the direct local service predict URL."""

    return _build_local_service_url(service_base_url, LOCAL_PREDICT_ROUTE)


def normalize_local_service_base_url(value: str) -> str:
    """Validate and normalize a direct local JointFM service base URL."""

    normalized_url = _normalize_non_whitespace_string(
        value,
        JOINTFM_LOCAL_BASE_URL_ENV,
    ).rstrip("/")
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise JointFMConfigurationError(
            f"{JOINTFM_LOCAL_BASE_URL_ENV} must be an http or https URL"
        )
    if not parsed_url.netloc:
        raise JointFMConfigurationError(
            f"{JOINTFM_LOCAL_BASE_URL_ENV} must include a hostname"
        )
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise JointFMConfigurationError(
            f"{JOINTFM_LOCAL_BASE_URL_ENV} must not include params, query, or fragment"
        )
    return normalized_url


def build_datarobot_prediction_headers(api_token: str) -> dict[str, str]:
    """Build hosted prediction headers using the notebook authorization scheme."""

    normalized_api_token = validate_datarobot_api_token(api_token)
    return {
        "Authorization": f"Bearer {normalized_api_token}",
        "Accept": "*/*",
        "Content-Type": "application/json;charset=UTF-8",
    }


def _merge_config_dotenv_and_environment(
    *,
    config: JointFMConfig,
    env: Mapping[str, str] | None,
    dotenv_path: str | Path | None,
) -> dict[str, str]:
    merged_values = config.deployment.to_environment_values(config.environment)
    if dotenv_path is not None:
        path = Path(dotenv_path)
        if path.exists():
            for key, value in dotenv_values(path).items():
                if value is not None:
                    merged_values[key] = value

    source_env = os.environ if env is None else env
    for key, value in source_env.items():
        merged_values[key] = value
    return merged_values


def _required_env(env: Mapping[str, str], name: str) -> str:
    try:
        value = env[name]
    except KeyError as exc:
        raise JointFMConfigurationError(f"{name} is required") from exc
    if value == "":
        raise JointFMConfigurationError(f"{name} must be non-empty")
    return value


def _resolve_single_deployment_selector(
    env: Mapping[str, str],
    environment: EnvironmentVariableConfig,
) -> str:
    deployment_selector_envs = environment.deployment_selector_names()
    selector_names = [
        selector_name
        for selector_name in deployment_selector_envs
        if selector_name in env and env[selector_name] != ""
    ]
    if len(selector_names) != 1:
        formatted_selectors = ", ".join(deployment_selector_envs)
        raise JointFMConfigurationError(
            f"Exactly one deployment selector is required: {formatted_selectors}"
        )
    return selector_names[0]


def _load_pulumi_target_outputs(
    outputs_path: str,
    deployment_target: str,
) -> Mapping[str, object]:
    normalized_target = _normalize_non_whitespace_string(
        deployment_target,
        JOINTFM_DEPLOYMENT_TARGET_ENV,
    )
    normalized_path = _normalize_non_whitespace_string(
        outputs_path,
        JOINTFM_PULUMI_OUTPUTS_PATH_ENV,
    )
    path = Path(normalized_path)
    if not path.is_file():
        raise JointFMConfigurationError(
            f"{JOINTFM_PULUMI_OUTPUTS_PATH_ENV} must reference a JSON file"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JointFMConfigurationError(
            f"{JOINTFM_PULUMI_OUTPUTS_PATH_ENV} must contain valid JSON"
        ) from exc
    except OSError as exc:
        raise JointFMConfigurationError(
            f"Unable to read {JOINTFM_PULUMI_OUTPUTS_PATH_ENV}"
        ) from exc

    if not isinstance(payload, dict):
        raise JointFMConfigurationError(
            f"{JOINTFM_PULUMI_OUTPUTS_PATH_ENV} must contain a JSON object"
        )
    if normalized_target not in payload:
        raise JointFMConfigurationError(
            f"{JOINTFM_PULUMI_OUTPUTS_PATH_ENV} is missing target {normalized_target!r}"
        )
    target_outputs = payload[normalized_target]
    if not isinstance(target_outputs, dict):
        raise JointFMConfigurationError("Pulumi target output must be a JSON object")
    return target_outputs


def _optional_string_output(outputs: Mapping[str, object], key: str) -> str | None:
    if key not in outputs:
        return None
    value = outputs[key]
    if not isinstance(value, str) or value == "":
        raise JointFMConfigurationError(f"Pulumi output {key!r} must be a string")
    return value


def _normalize_absolute_https_url(value: str, env_name: str) -> str:
    normalized_url = _normalize_non_whitespace_string(value, env_name).rstrip("/")
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme != "https":
        raise JointFMConfigurationError(f"{env_name} must be an https URL")
    if not parsed_url.netloc:
        raise JointFMConfigurationError(f"{env_name} must include a hostname")
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise JointFMConfigurationError(
            f"{env_name} must not include params, query, or fragment"
        )
    return normalized_url


def _build_local_service_url(service_base_url: str, route: str) -> str:
    return f"{normalize_local_service_base_url(service_base_url)}{route}"


def _normalize_non_whitespace_string(value: str, name: str) -> str:
    if value == "":
        raise JointFMConfigurationError(f"{name} must be non-empty")
    if value.strip() != value or any(character.isspace() for character in value):
        raise JointFMConfigurationError(f"{name} must not contain whitespace")
    return value
