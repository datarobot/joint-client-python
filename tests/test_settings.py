"""Tests for the settings surface of jointfm_client."""

import json

import pytest

from jointfm_client import (
    DATAROBOT_API_TOKEN_ENV,
    DATAROBOT_ENDPOINT_ENV,
    JOINTFM_DEPLOYMENT_ID_ENV,
    JOINTFM_DEPLOYMENT_TARGET_ENV,
    JOINTFM_DEPLOYMENT_URL_ENV,
    JOINTFM_LOCAL_BASE_URL_ENV,
    JOINTFM_MODEL_VERSION_ENV,
    JOINTFM_PREDICT_URL_ENV,
    JOINTFM_PULUMI_OUTPUTS_PATH_ENV,
    JOINTFM_SCHEMA_VERSION_ENV,
    JointFMClient,
    JointFMConfigurationError,
    build_datarobot_prediction_headers,
    build_hosted_predict_url,
    build_hosted_predict_url_from_deployment_url,
    build_local_health_url,
    build_local_predict_url,
    deployment_id_from_hosted_deployment_url,
    deployment_url_from_hosted_predict_url,
    load_settings,
    normalize_datarobot_endpoint,
    normalize_deployment_id,
    normalize_hosted_deployment_url,
    normalize_hosted_predict_url,
    validate_datarobot_api_token,
)


def _hosted_env(**overrides: str) -> dict[str, str]:
    """Hosted env."""
    env = {
        DATAROBOT_ENDPOINT_ENV: "https://app.datarobot.com/api/v2/",
        DATAROBOT_API_TOKEN_ENV: "secret-token",
        JOINTFM_DEPLOYMENT_ID_ENV: "deployment-id",
        JOINTFM_SCHEMA_VERSION_ENV: "v1",
        JOINTFM_MODEL_VERSION_ENV: "jointfm-inference:0.2.0+ckpt.sdk-test",
    }
    env.update(overrides)
    return env


def test_load_settings_from_environment_with_deployment_id_builds_hosted_url() -> None:
    """Load settings from environment with deployment id builds hosted url."""
    settings = load_settings(env=_hosted_env(), dotenv_path=None)

    assert settings.datarobot_endpoint == "https://app.datarobot.com/api/v2"
    assert settings.schema_version == "v1"
    assert settings.model_version == "jointfm-inference:0.2.0+ckpt.sdk-test"
    assert settings.deployment_id == "deployment-id"
    assert settings.predict_url == (
        "https://app.datarobot.com/api/v2/deployments/"
        "deployment-id/predictionsUnstructured"
    )
    assert settings.health_url == settings.predict_url
    assert "secret-token" not in repr(settings)


def test_load_settings_with_local_service_base_url_builds_direct_urls() -> None:
    """Load settings with local service base url builds direct urls."""
    settings = load_settings(
        env={
            JOINTFM_LOCAL_BASE_URL_ENV: "http://127.0.0.1:8080/",
            JOINTFM_SCHEMA_VERSION_ENV: "v1",
            JOINTFM_MODEL_VERSION_ENV: "jointfm-inference:0.2.0+ckpt.local-test",
        },
        dotenv_path=None,
    )

    assert settings.deployment_selector == "local_service"
    assert settings.datarobot_endpoint is None
    assert settings.datarobot_api_token is None
    assert settings.deployment_id is None
    assert settings.deployment_url is None
    assert settings.local_base_url == "http://127.0.0.1:8080"
    assert settings.health_url == "http://127.0.0.1:8080/healthz"
    assert settings.predict_url == "http://127.0.0.1:8080/predict"


def test_load_settings_with_deployment_url_selector_builds_prediction_url() -> None:
    """Load settings with deployment url selector builds prediction url."""
    env = _hosted_env(
        **{
            JOINTFM_DEPLOYMENT_URL_ENV: (
                "https://app.datarobot.com/api/v2/deployments/deployment-from-url"
            )
        }
    )
    del env[JOINTFM_DEPLOYMENT_ID_ENV]

    settings = load_settings(env=env, dotenv_path=None)

    assert settings.deployment_selector == "deployment_url"
    assert settings.deployment_id == "deployment-from-url"
    assert settings.deployment_url == (
        "https://app.datarobot.com/api/v2/deployments/deployment-from-url"
    )
    assert settings.predict_url == (
        "https://app.datarobot.com/api/v2/deployments/"
        "deployment-from-url/predictionsUnstructured"
    )
    assert settings.health_url == settings.predict_url


def test_load_settings_with_predict_url_selector_finds_deployment_url() -> None:
    """Load settings with predict url selector finds deployment url."""
    env = _hosted_env(
        **{
            JOINTFM_PREDICT_URL_ENV: (
                "https://app.datarobot.com/api/v2/deployments/"
                "deployment-from-predict/predictionsUnstructured"
            )
        }
    )
    del env[JOINTFM_DEPLOYMENT_ID_ENV]

    settings = load_settings(env=env, dotenv_path=None)

    assert settings.deployment_selector == "predict_url"
    assert settings.deployment_id == "deployment-from-predict"
    assert settings.deployment_url == (
        "https://app.datarobot.com/api/v2/deployments/deployment-from-predict"
    )
    assert settings.predict_url == (
        "https://app.datarobot.com/api/v2/deployments/"
        "deployment-from-predict/predictionsUnstructured"
    )
    assert settings.health_url == settings.predict_url


def test_load_settings_reads_dotenv_without_overriding_environment(tmp_path) -> None:
    """Load settings reads dotenv without overriding environment."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN=file-token",
                "JOINTFM_DEPLOYMENT_ID=file-deployment-id",
                "JOINTFM_SCHEMA_VERSION=v1",
                "JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.sdk-test",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        env={DATAROBOT_API_TOKEN_ENV: "env-token"},
        dotenv_path=dotenv_path,
    )

    assert settings.datarobot_api_token == "env-token"
    assert settings.deployment_id == "file-deployment-id"


def test_load_settings_requires_explicit_schema_version() -> None:
    """Load settings requires explicit schema version."""
    env_without_schema_version = _hosted_env()
    del env_without_schema_version[JOINTFM_SCHEMA_VERSION_ENV]

    with pytest.raises(JointFMConfigurationError, match=JOINTFM_SCHEMA_VERSION_ENV):
        load_settings(env=env_without_schema_version, dotenv_path=None)


def test_load_settings_makes_model_version_optional() -> None:
    """Load settings makes model version optional."""
    env_without_model_version = _hosted_env()
    del env_without_model_version[JOINTFM_MODEL_VERSION_ENV]

    settings = load_settings(env=env_without_model_version, dotenv_path=None)

    assert settings.model_version is None


def test_load_settings_rejects_unsupported_schema_version() -> None:
    """Load settings rejects unsupported schema version."""
    with pytest.raises(JointFMConfigurationError, match=JOINTFM_SCHEMA_VERSION_ENV):
        load_settings(
            env=_hosted_env(**{JOINTFM_SCHEMA_VERSION_ENV: "v2"}),
            dotenv_path=None,
        )


def test_load_settings_rejects_missing_credentials_without_defaults() -> None:
    """Load settings rejects missing credentials without defaults."""
    with pytest.raises(JointFMConfigurationError, match=DATAROBOT_ENDPOINT_ENV):
        load_settings(
            env={
                DATAROBOT_API_TOKEN_ENV: "secret-token",
                JOINTFM_DEPLOYMENT_ID_ENV: "deployment-id",
                JOINTFM_SCHEMA_VERSION_ENV: "v1",
                JOINTFM_MODEL_VERSION_ENV: "jointfm-inference:0.2.0+ckpt.sdk-test",
            },
            dotenv_path=None,
        )

    with pytest.raises(JointFMConfigurationError, match=DATAROBOT_API_TOKEN_ENV):
        load_settings(
            env={
                DATAROBOT_ENDPOINT_ENV: "https://app.datarobot.com/api/v2",
                JOINTFM_DEPLOYMENT_ID_ENV: "deployment-id",
                JOINTFM_SCHEMA_VERSION_ENV: "v1",
                JOINTFM_MODEL_VERSION_ENV: "jointfm-inference:0.2.0+ckpt.sdk-test",
            },
            dotenv_path=None,
        )


@pytest.mark.parametrize(
    ("endpoint", "token"),
    [
        ("http://app.datarobot.com/api/v2", "secret-token"),
        ("https://app.datarobot.com/api/v1", "secret-token"),
        ("https://app.datarobot.com/api/v2", "secret token"),
    ],
)
def test_load_settings_rejects_malformed_credentials(endpoint: str, token: str) -> None:
    """Load settings rejects malformed credentials."""
    with pytest.raises(JointFMConfigurationError):
        load_settings(
            env=_hosted_env(
                **{
                    DATAROBOT_ENDPOINT_ENV: endpoint,
                    DATAROBOT_API_TOKEN_ENV: token,
                }
            ),
            dotenv_path=None,
        )


def test_load_settings_requires_exactly_one_deployment_selector() -> None:
    """Load settings requires exactly one deployment selector."""
    env_without_selector = _hosted_env()
    del env_without_selector[JOINTFM_DEPLOYMENT_ID_ENV]

    with pytest.raises(JointFMConfigurationError, match="deployment selector"):
        load_settings(env=env_without_selector, dotenv_path=None)

    with pytest.raises(JointFMConfigurationError, match="deployment selector"):
        load_settings(
            env=_hosted_env(
                **{
                    JOINTFM_PREDICT_URL_ENV: (
                        "https://app.datarobot.com/api/v2/deployments/"
                        "other-deployment-id/predictionsUnstructured"
                    )
                }
            ),
            dotenv_path=None,
        )

    with pytest.raises(JointFMConfigurationError, match="deployment selector"):
        load_settings(
            env=_hosted_env(**{JOINTFM_LOCAL_BASE_URL_ENV: "http://127.0.0.1:8080"}),
            dotenv_path=None,
        )


def test_load_settings_resolves_named_pulumi_target_from_saved_outputs(
    tmp_path,
) -> None:
    """Load settings resolves named pulumi target from saved outputs."""
    outputs_path = tmp_path / "jointfm-pulumi-outputs.json"
    outputs_path.write_text(
        json.dumps({"fin-studentt": {"deployment_id": "pulumi-deployment-id"}}),
        encoding="utf-8",
    )
    env = _hosted_env(
        **{
            JOINTFM_DEPLOYMENT_TARGET_ENV: "fin-studentt",
            JOINTFM_PULUMI_OUTPUTS_PATH_ENV: str(outputs_path),
        }
    )
    del env[JOINTFM_DEPLOYMENT_ID_ENV]

    settings = load_settings(env=env, dotenv_path=None)

    assert settings.deployment_target == "fin-studentt"
    assert settings.deployment_id == "pulumi-deployment-id"
    assert settings.predict_url == (
        "https://app.datarobot.com/api/v2/deployments/"
        "pulumi-deployment-id/predictionsUnstructured"
    )


@pytest.mark.parametrize(
    ("target_outputs", "expected_deployment_id"),
    [
        (
            {
                "deployment_url": (
                    "https://app.datarobot.com/api/v2/deployments/"
                    "pulumi-deployment-url-id"
                )
            },
            "pulumi-deployment-url-id",
        ),
        (
            {
                "predict_url": (
                    "https://app.datarobot.com/api/v2/deployments/"
                    "pulumi-predict-url-id/predictionsUnstructured"
                )
            },
            "pulumi-predict-url-id",
        ),
    ],
)
def test_load_settings_resolves_named_pulumi_target_url_outputs(
    tmp_path,
    target_outputs: dict[str, str],
    expected_deployment_id: str,
) -> None:
    """Load settings resolves named pulumi target url outputs."""
    outputs_path = tmp_path / "jointfm-pulumi-outputs.json"
    outputs_path.write_text(
        json.dumps({"fin-studentt": target_outputs}),
        encoding="utf-8",
    )
    env = _hosted_env(
        **{
            JOINTFM_DEPLOYMENT_TARGET_ENV: "fin-studentt",
            JOINTFM_PULUMI_OUTPUTS_PATH_ENV: str(outputs_path),
        }
    )
    del env[JOINTFM_DEPLOYMENT_ID_ENV]

    settings = load_settings(env=env, dotenv_path=None)

    assert settings.deployment_selector == "pulumi_target"
    assert settings.deployment_target == "fin-studentt"
    assert settings.deployment_id == expected_deployment_id
    assert settings.predict_url == (
        "https://app.datarobot.com/api/v2/deployments/"
        f"{expected_deployment_id}/predictionsUnstructured"
    )


def test_load_settings_rejects_pulumi_target_without_url_outputs(tmp_path) -> None:
    """Load settings rejects pulumi target without url outputs."""
    outputs_path = tmp_path / "jointfm-pulumi-outputs.json"
    outputs_path.write_text(json.dumps({"fin-studentt": {}}), encoding="utf-8")
    env = _hosted_env(
        **{
            JOINTFM_DEPLOYMENT_TARGET_ENV: "fin-studentt",
            JOINTFM_PULUMI_OUTPUTS_PATH_ENV: str(outputs_path),
        }
    )
    del env[JOINTFM_DEPLOYMENT_ID_ENV]

    with pytest.raises(JointFMConfigurationError, match="deployment_id"):
        load_settings(env=env, dotenv_path=None)


def test_load_settings_rejects_invalid_pulumi_outputs(tmp_path) -> None:
    """Load settings rejects invalid pulumi outputs."""
    missing_outputs_path = tmp_path / "missing.json"
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{", encoding="utf-8")
    list_payload_path = tmp_path / "list.json"
    list_payload_path.write_text("[]", encoding="utf-8")
    missing_target_path = tmp_path / "missing-target.json"
    missing_target_path.write_text(json.dumps({"other": {}}), encoding="utf-8")
    non_object_target_path = tmp_path / "non-object-target.json"
    non_object_target_path.write_text(
        json.dumps({"fin-studentt": []}), encoding="utf-8"
    )
    invalid_output_path = tmp_path / "invalid-output.json"
    invalid_output_path.write_text(
        json.dumps({"fin-studentt": {"deployment_id": ""}}),
        encoding="utf-8",
    )

    invalid_cases = [
        (missing_outputs_path, "must reference a JSON file"),
        (invalid_json_path, "valid JSON"),
        (list_payload_path, "JSON object"),
        (missing_target_path, "missing target"),
        (non_object_target_path, "JSON object"),
        (invalid_output_path, "must be a string"),
    ]

    for outputs_path, expected_message in invalid_cases:
        env = _hosted_env(
            **{
                JOINTFM_DEPLOYMENT_TARGET_ENV: "fin-studentt",
                JOINTFM_PULUMI_OUTPUTS_PATH_ENV: str(outputs_path),
            }
        )
        del env[JOINTFM_DEPLOYMENT_ID_ENV]

        with pytest.raises(JointFMConfigurationError, match=expected_message):
            load_settings(env=env, dotenv_path=None)


def test_hosted_and_local_url_builders_are_separate() -> None:
    """Hosted and local url builders are separate."""
    assert build_hosted_predict_url(
        "https://app.datarobot.com/api/v2/",
        "deployment-id",
    ) == (
        "https://app.datarobot.com/api/v2/deployments/"
        "deployment-id/predictionsUnstructured"
    )
    assert build_local_health_url("http://localhost:8080/") == (
        "http://localhost:8080/healthz"
    )
    assert build_local_predict_url("http://localhost:8080/") == (
        "http://localhost:8080/predict"
    )


def test_hosted_url_helpers_normalize_supported_forms() -> None:
    """Hosted url helpers normalize supported forms."""
    deployment_url = "https://app.datarobot.com/api/v2/deployments/deployment-id"
    predict_url = f"{deployment_url}/predictionsUnstructured"

    assert normalize_hosted_deployment_url(f"{deployment_url}/") == deployment_url
    assert normalize_hosted_predict_url(f"{predict_url}/") == predict_url
    assert build_hosted_predict_url_from_deployment_url(deployment_url) == predict_url
    assert deployment_id_from_hosted_deployment_url(deployment_url) == "deployment-id"
    assert deployment_url_from_hosted_predict_url(predict_url) == deployment_url


def test_url_validators_reject_ambiguous_or_malformed_values() -> None:
    """Url validators reject ambiguous or malformed values."""
    with pytest.raises(JointFMConfigurationError, match="hostname"):
        normalize_datarobot_endpoint("https:///api/v2")

    with pytest.raises(JointFMConfigurationError, match="params, query, or fragment"):
        normalize_datarobot_endpoint("https://app.datarobot.com/api/v2?extra=true")

    with pytest.raises(JointFMConfigurationError, match="deployment ID"):
        normalize_deployment_id("deployments/deployment-id")

    with pytest.raises(JointFMConfigurationError, match="predictionsUnstructured"):
        normalize_hosted_predict_url(
            "https://app.datarobot.com/api/v2/deployments/deployment-id"
        )

    with pytest.raises(JointFMConfigurationError, match="deployments"):
        deployment_id_from_hosted_deployment_url(
            "https://app.datarobot.com/api/v2/not-deployments/deployment-id"
        )

    with pytest.raises(JointFMConfigurationError, match="https URL"):
        normalize_hosted_deployment_url(
            "http://app.datarobot.com/api/v2/deployments/deployment-id"
        )

    with pytest.raises(JointFMConfigurationError, match="hostname"):
        normalize_hosted_deployment_url("https:///api/v2/deployments/deployment-id")

    with pytest.raises(JointFMConfigurationError, match="params, query, or fragment"):
        normalize_hosted_deployment_url(
            "https://app.datarobot.com/api/v2/deployments/deployment-id?extra=true"
        )

    with pytest.raises(JointFMConfigurationError, match="http or https"):
        build_local_health_url("ftp://localhost:8080")

    with pytest.raises(JointFMConfigurationError, match="hostname"):
        build_local_predict_url("http:///predict")

    with pytest.raises(JointFMConfigurationError, match="non-empty"):
        validate_datarobot_api_token("")


def test_datarobot_prediction_headers_use_notebook_authorization_scheme() -> None:
    """Datarobot prediction headers use notebook authorization scheme."""
    assert build_datarobot_prediction_headers("secret-token") == {
        "Authorization": "Bearer secret-token",
        "Accept": "*/*",
        "Content-Type": "application/json;charset=UTF-8",
    }


def test_jointfm_client_from_env_attaches_non_secret_settings() -> None:
    """Jointfm client from env attaches non secret settings."""
    client = JointFMClient.from_env(env=_hosted_env(), dotenv_path=None)

    assert client.settings is not None
    assert client.settings.predict_url.endswith("/predictionsUnstructured")
    assert "secret-token" not in repr(client)
