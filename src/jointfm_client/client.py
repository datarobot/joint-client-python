# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Public synchronous client shape for JointFM V1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, Self, cast
from urllib.parse import urlparse

from jointfm_client.adapters import build_forecast_payload_from_dataframe
from jointfm_client.configuration import (
    DATAROBOT_REQUEST_ID_HEADERS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS,
    JointFMConfig,
    load_configuration,
)
from jointfm_client.contract import (
    DEFAULT_CALENDAR_ID,
    HEALTH_REQUEST_TYPE,
    SCHEMA_VERSION,
    ColumnRole,
    ColumnSpec,
    DataFrameSchema,
    HealthMetadata,
    ReturnMode,
    TimeIndexMode,
    TimeValueKind,
    build_forecast_payload,
    validate_service_metadata,
)
from jointfm_client.exceptions import (
    JointFMConfigurationError,
    JointFMHTTPStatusError,
    JointFMServiceError,
    UnsupportedModelVersionError,
)
from jointfm_client.contract import (
    ForecastDiagnostics,
    ForecastResponse,
    MeanForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from jointfm_client.settings import (
    JointFMSettings,
    load_settings,
    validate_jointfm_model_version,
)
from jointfm_client.transport import (
    JSONTransport,
    JointFMHTTPTransport,
    JointFMRetryConfig,
    JointFMTimeoutConfig,
)

_SAMPLE_CAP_ERROR_PATTERN = re.compile(
    r"n_samples exceeds the configured container cap:\s*"
    r"requested\s+(?P<requested>[0-9]+),\s*max\s+(?P<cap>[0-9]+)"
)


class JointFMClient:
    """Synchronous entrypoint for low-level predictions and forecast helpers."""

    def __init__(
        self,
        *,
        settings: JointFMSettings | None = None,
        health_url: str | None = None,
        predict_url: str | None = None,
        transport: JSONTransport | None = None,
        timeout: JointFMTimeoutConfig = JointFMTimeoutConfig(),
        retry_config: JointFMRetryConfig = JointFMRetryConfig(),
        response_body_excerpt_characters: int = DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS,
        datarobot_request_id_headers: Sequence[str] = DATAROBOT_REQUEST_ID_HEADERS,
    ) -> None:
        self.settings = settings
        self.health_url = _configured_url(
            "health_url",
            configured_url=None if settings is None else settings.health_url,
            explicit_url=health_url,
        )
        self.predict_url = _configured_url(
            "predict_url",
            configured_url=None if settings is None else settings.predict_url,
            explicit_url=predict_url,
        )
        self._transport = transport
        self._timeout = timeout
        self._retry_config = retry_config
        self._response_body_excerpt_characters = response_body_excerpt_characters
        self._datarobot_request_id_headers = datarobot_request_id_headers
        self._health_metadata: HealthMetadata | None = None
        self._sample_batch_cap: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        dotenv_path: str | Path | None = ".env",
        config_path: str | Path | None = DEFAULT_CONFIG_PATH,
        config: JointFMConfig | Mapping[str, Any] | None = None,
        timeout: JointFMTimeoutConfig | None = None,
        retry_config: JointFMRetryConfig | None = None,
        response_body_excerpt_characters: int | None = None,
        datarobot_request_id_headers: Sequence[str] | None = None,
    ) -> Self:
        """Create a client from environment variables and an optional `.env` file."""
        jointfm_config = load_configuration(config_path=config_path, overrides=config)
        return cls(
            settings=load_settings(
                env=env,
                dotenv_path=dotenv_path,
                config_path=None,
                config=jointfm_config,
            ),
            timeout=_resolve_timeout_config(timeout, jointfm_config),
            retry_config=_resolve_retry_config(retry_config, jointfm_config),
            response_body_excerpt_characters=_resolve_response_body_excerpt_characters(
                response_body_excerpt_characters,
                jointfm_config,
            ),
            datarobot_request_id_headers=_resolve_datarobot_request_id_headers(
                datarobot_request_id_headers,
                jointfm_config,
            ),
        )

    def health(self, *, cache: bool = False, refresh: bool = False) -> HealthMetadata:
        """Return service metadata from the configured JointFM endpoint.

        Local deployments (``deployment_selector == "local_service"``) probe the
        container's ``GET /healthz`` route directly. Hosted DataRobot deployments
        POST ``{"request_type": "health"}`` to ``predict_url`` because DataRobot's
        deployment gateway only proxies the unstructured prediction route; the
        container short-circuits that body before any schema or model version
        validation and returns the same typed health payload.
        """
        if cache and not refresh and self._health_metadata is not None:
            return self._health_metadata

        if self._uses_predict_route_for_health():
            payload = self._fetch_hosted_health_payload()
        else:
            health_url = self._require_health_url()
            payload = self._transport_for_request().get_json(health_url)
        expected_model_version = (
            None if self.settings is None else self.settings.model_version
        )
        validate_service_metadata(
            payload, expected_model_version=expected_model_version
        )
        metadata = HealthMetadata.from_payload(payload)
        if cache:
            self._health_metadata = metadata
        return metadata

    def _uses_predict_route_for_health(self) -> bool:
        """Return whether hosted health probes must POST to the predict route."""
        if self.settings is None:
            return False
        return self.settings.deployment_selector != "local_service"

    def _fetch_hosted_health_payload(self) -> Mapping[str, Any]:
        """POST a minimal health discriminator to the hosted predict URL."""
        predict_url = self._require_predict_url("health")
        return self._transport_for_request().post_json(
            predict_url,
            {"request_type": HEALTH_REQUEST_TYPE},
        )

    def predict(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Submit one V1 JSON prediction payload to the configured endpoint."""
        predict_url = self._require_predict_url("predict")
        model_version = payload.get("model_version")
        if not isinstance(model_version, str):
            raise JointFMConfigurationError(
                "JointFMClient.predict() requires payload['model_version']"
            )
        self._resolve_model_version(model_version=model_version)
        response_payload = self._transport_for_request().post_json(predict_url, payload)
        ForecastResponse.raise_for_errors(response_payload)
        return response_payload

    def forecast(
        self,
        history: Any,
        *,
        query_times: Sequence[Any],
        schema: DataFrameSchema | None = None,
        time_index_mode: TimeIndexMode = "ordinal",
        columns: Sequence[ColumnSpec] | None = None,
        time_column: str | None = None,
        requested_columns: Sequence[str | int] | None = None,
        return_mode: ReturnMode = "mean",
        model_version: str | None = None,
        n_samples: int | None = None,
        quantiles: Sequence[float | int] | None = None,
        seed: int | None = None,
        schema_version: str | None = None,
        time_scale_seconds: float | int | None = None,
        use_local_normalized_time: bool = False,
        calendar_id: str = DEFAULT_CALENDAR_ID,
        timezone: str | None = None,
        target_columns: Sequence[str] | None = None,
        feature_columns: Sequence[str] | None = None,
        known_dynamic_columns: Sequence[str] | None = None,
        past_dynamic_columns: Sequence[str] | None = None,
        static_columns: Sequence[str] | None = None,
        column_roles: Mapping[str, ColumnRole] | None = None,
        categorical_columns: Sequence[str] | None = None,
        categorical_mappings: Mapping[str, Mapping[str | int, int]] | None = None,
        ordinal_columns: Sequence[str] | None = None,
        ordinal_mappings: Mapping[str, Mapping[str | int, int]] | None = None,
        count_columns: Sequence[str] | None = None,
        time_value_columns: Sequence[str] | Mapping[str, TimeValueKind] | None = None,
        nullable_columns: Sequence[str] | None = None,
        bounds: Mapping[str, tuple[float | int | None, float | int | None]]
        | None = None,
    ) -> ForecastResponse:
        """Build and submit a forecast request from tabular history inputs."""
        predict_url = self._require_predict_url("forecast")
        resolved_model_version = self._resolve_model_version(
            model_version=model_version,
        )
        resolved_schema_version = self._resolve_schema_version(schema_version)
        if schema is not None or _is_history_row_sequence(history):
            payload = self._forecast_payload_from_rows(
                history,
                schema=schema,
                time_index_mode=time_index_mode,
                columns=columns,
                time_column=time_column,
                query_times=query_times,
                requested_columns=requested_columns,
                return_mode=return_mode,
                model_version=resolved_model_version,
                n_samples=n_samples,
                quantiles=quantiles,
                seed=seed,
                schema_version=resolved_schema_version,
                time_scale_seconds=time_scale_seconds,
                use_local_normalized_time=use_local_normalized_time,
                calendar_id=calendar_id,
                timezone=timezone,
            )
        else:
            payload = build_forecast_payload_from_dataframe(
                history,
                model_version=resolved_model_version,
                time_index_mode=time_index_mode,
                query_times=query_times,
                columns=columns,
                time_column=time_column,
                requested_columns=requested_columns,
                return_mode=return_mode,
                n_samples=n_samples,
                quantiles=quantiles,
                seed=seed,
                schema_version=resolved_schema_version,
                time_scale_seconds=time_scale_seconds,
                use_local_normalized_time=use_local_normalized_time,
                calendar_id=calendar_id,
                timezone=timezone,
                target_columns=target_columns,
                feature_columns=feature_columns,
                known_dynamic_columns=known_dynamic_columns,
                past_dynamic_columns=past_dynamic_columns,
                static_columns=static_columns,
                column_roles=column_roles,
                categorical_columns=categorical_columns,
                categorical_mappings=categorical_mappings,
                ordinal_columns=ordinal_columns,
                ordinal_mappings=ordinal_mappings,
                count_columns=count_columns,
                time_value_columns=time_value_columns,
                nullable_columns=nullable_columns,
                bounds=bounds,
            )
        sample_cap = _known_sample_batch_cap(payload, self._sample_batch_cap)
        if sample_cap is not None:
            return self._forecast_sample_batches(predict_url, payload, sample_cap)

        try:
            response_payload = self._transport_for_request().post_json(
                predict_url, payload
            )
        except JointFMHTTPStatusError as error:
            sample_cap = _sample_batch_cap_from_error(error, payload)
            if sample_cap is None:
                raise
            self._sample_batch_cap = sample_cap
            return self._forecast_sample_batches(predict_url, payload, sample_cap)

        return _forecast_response_from_payload(response_payload, payload)

    def forecast_mean(
        self,
        history: Any,
        *,
        query_times: Sequence[Any],
        schema: DataFrameSchema | None = None,
        time_index_mode: TimeIndexMode = "ordinal",
        columns: Sequence[ColumnSpec] | None = None,
        time_column: str | None = None,
        requested_columns: Sequence[str | int] | None = None,
        model_version: str | None = None,
        seed: int | None = None,
    ) -> MeanForecastResult:
        """Forecast mean values through the shared forecast validation path."""
        return cast(
            MeanForecastResult,
            self.forecast(
                history,
                query_times=query_times,
                schema=schema,
                time_index_mode=time_index_mode,
                columns=columns,
                time_column=time_column,
                requested_columns=requested_columns,
                return_mode="mean",
                model_version=model_version,
                seed=seed,
            ),
        )

    def forecast_samples(
        self,
        history: Any,
        *,
        query_times: Sequence[Any],
        schema: DataFrameSchema | None = None,
        time_index_mode: TimeIndexMode = "ordinal",
        columns: Sequence[ColumnSpec] | None = None,
        time_column: str | None = None,
        requested_columns: Sequence[str | int] | None = None,
        model_version: str | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> SampleForecastResult:
        """Forecast sample paths through the shared forecast validation path."""
        return cast(
            SampleForecastResult,
            self.forecast(
                history,
                query_times=query_times,
                schema=schema,
                time_index_mode=time_index_mode,
                columns=columns,
                time_column=time_column,
                requested_columns=requested_columns,
                return_mode="samples",
                model_version=model_version,
                n_samples=n_samples,
                seed=seed,
            ),
        )

    def forecast_quantiles(
        self,
        history: Any,
        *,
        query_times: Sequence[Any],
        schema: DataFrameSchema | None = None,
        time_index_mode: TimeIndexMode = "ordinal",
        columns: Sequence[ColumnSpec] | None = None,
        time_column: str | None = None,
        requested_columns: Sequence[str | int] | None = None,
        model_version: str | None = None,
        n_samples: int | None = None,
        quantiles: Sequence[float | int] | None = None,
        seed: int | None = None,
    ) -> QuantileForecastResult:
        """Forecast quantiles through the shared forecast validation path."""
        return cast(
            QuantileForecastResult,
            self.forecast(
                history,
                query_times=query_times,
                schema=schema,
                time_index_mode=time_index_mode,
                columns=columns,
                time_column=time_column,
                requested_columns=requested_columns,
                return_mode="quantiles",
                model_version=model_version,
                n_samples=n_samples,
                quantiles=quantiles,
                seed=seed,
            ),
        )

    def refresh_health(self, *, cache: bool = True) -> HealthMetadata:
        """Fetch fresh service metadata and update the explicit cache by default."""
        return self.health(cache=cache, refresh=True)

    def _forecast_payload_from_rows(
        self,
        history: Any,
        *,
        schema: DataFrameSchema | None,
        time_index_mode: TimeIndexMode,
        columns: Sequence[ColumnSpec] | None,
        time_column: str | None,
        query_times: Sequence[Any],
        requested_columns: Sequence[str | int] | None,
        return_mode: ReturnMode,
        model_version: str,
        n_samples: int | None,
        quantiles: Sequence[float | int] | None,
        seed: int | None,
        schema_version: str,
        time_scale_seconds: float | int | None,
        use_local_normalized_time: bool,
        calendar_id: str,
        timezone: str | None,
    ) -> dict[str, Any]:
        if schema is None:
            if columns is None:
                raise ValueError(
                    "columns or schema is required for row-sequence forecasts"
                )
            schema = DataFrameSchema(
                columns=columns,
                time_index_mode=time_index_mode,
                time_column=time_column,
                time_scale_seconds=time_scale_seconds,
                use_local_normalized_time=use_local_normalized_time,
                calendar_id=calendar_id,
                timezone=timezone,
            )
        return build_forecast_payload(
            model_version=model_version,
            schema=schema,
            history_rows=cast(Sequence[Mapping[str, Any]], history),
            query_times=query_times,
            requested_columns=requested_columns,
            return_mode=return_mode,
            n_samples=n_samples,
            quantiles=quantiles,
            seed=seed,
            schema_version=schema_version,
        )

    def _forecast_sample_batches(
        self,
        predict_url: str,
        payload: Mapping[str, Any],
        sample_cap: int,
    ) -> SampleForecastResult:
        requested_samples = cast(int, payload["n_samples"])
        remaining_samples = requested_samples
        batch_index = 0
        batch_results: list[SampleForecastResult] = []

        while remaining_samples > 0:
            batch_samples = min(sample_cap, remaining_samples)
            batch_payload = dict(payload)
            batch_payload["n_samples"] = batch_samples
            _set_batch_seed(batch_payload, batch_index)
            response_payload = self._transport_for_request().post_json(
                predict_url,
                batch_payload,
            )
            batch_result = _forecast_response_from_payload(
                response_payload,
                batch_payload,
            )
            if not isinstance(batch_result, SampleForecastResult):
                raise JointFMServiceError(
                    "JointFM forecast response violated the V1 contract: "
                    "sample batching requires sample forecast responses"
                )
            batch_results.append(batch_result)
            remaining_samples -= batch_samples
            batch_index += 1

        try:
            return _merge_sample_forecast_results(batch_results, payload)
        except ValueError as error:
            raise JointFMServiceError(
                f"JointFM forecast response violated the V1 contract: {error}"
            ) from error

    def _require_settings(self, method_name: str) -> JointFMSettings:
        if self.settings is None:
            raise JointFMConfigurationError(
                f"JointFMClient.{method_name}() requires settings; use from_env() or pass settings"
            )
        return self.settings

    def _require_health_url(self) -> str:
        if self.health_url is None:
            raise JointFMConfigurationError(
                "JointFMClient.health() requires settings or health_url"
            )
        return self.health_url

    def _require_predict_url(self, method_name: str) -> str:
        if self.predict_url is None:
            raise JointFMConfigurationError(
                f"JointFMClient.{method_name}() requires settings or predict_url"
            )
        return self.predict_url

    def _transport_for_request(self) -> JSONTransport:
        if self._transport is None:
            if self.settings is None:
                self._transport = JointFMHTTPTransport(
                    timeout=self._timeout,
                    retry_config=self._retry_config,
                    response_body_excerpt_characters=self._response_body_excerpt_characters,
                    datarobot_request_id_headers=self._datarobot_request_id_headers,
                )
            else:
                self._transport = JointFMHTTPTransport.from_settings(
                    self.settings,
                    timeout=self._timeout,
                    retry_config=self._retry_config,
                    response_body_excerpt_characters=self._response_body_excerpt_characters,
                    datarobot_request_id_headers=self._datarobot_request_id_headers,
                )
        return self._transport

    def _resolve_model_version(
        self,
        *,
        model_version: str | None,
    ) -> str:
        configured_model_version = model_version
        if configured_model_version is None and self.settings is not None:
            configured_model_version = self.settings.model_version
        if configured_model_version is None:
            if self._health_metadata is None:
                self.health(cache=True)
            assert self._health_metadata is not None
            return self._health_metadata.model_version

        normalized_model_version = validate_jointfm_model_version(
            configured_model_version
        )
        if (
            self._health_metadata is not None
            and normalized_model_version != self._health_metadata.model_version
        ):
            raise UnsupportedModelVersionError(
                "Unsupported JointFM model_version: "
                f"expected {self._health_metadata.model_version!r}, got {normalized_model_version!r}"
            )
        return normalized_model_version

    def _resolve_schema_version(self, schema_version: str | None) -> str:
        if schema_version is not None:
            return schema_version
        if self.settings is not None:
            return self.settings.schema_version
        return SCHEMA_VERSION


def _configured_url(
    field: str,
    *,
    configured_url: str | None,
    explicit_url: str | None,
) -> str | None:
    normalized_configured_url = (
        None if configured_url is None else _normalize_client_url(configured_url, field)
    )
    normalized_explicit_url = (
        None if explicit_url is None else _normalize_client_url(explicit_url, field)
    )
    if (
        normalized_configured_url is not None
        and normalized_explicit_url is not None
        and normalized_configured_url != normalized_explicit_url
    ):
        raise JointFMConfigurationError(
            f"JointFMClient {field} conflicts with settings.{field}"
        )
    return normalized_explicit_url or normalized_configured_url


def _normalize_client_url(value: str, field: str) -> str:
    if value == "":
        raise JointFMConfigurationError(f"{field} must be non-empty")
    if value.strip() != value or any(character.isspace() for character in value):
        raise JointFMConfigurationError(f"{field} must not contain whitespace")
    normalized_url = value.rstrip("/")
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise JointFMConfigurationError(f"{field} must be an http or https URL")
    if not parsed_url.netloc:
        raise JointFMConfigurationError(f"{field} must include a hostname")
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise JointFMConfigurationError(
            f"{field} must not include params, query, or fragment"
        )
    return normalized_url


def _resolve_timeout_config(
    timeout: JointFMTimeoutConfig | None,
    config: JointFMConfig,
) -> JointFMTimeoutConfig:
    if timeout is not None:
        return timeout
    return JointFMTimeoutConfig(
        connect_seconds=config.transport.timeout.connect_seconds,
        read_seconds=config.transport.timeout.read_seconds,
    )


def _resolve_retry_config(
    retry_config: JointFMRetryConfig | None,
    config: JointFMConfig,
) -> JointFMRetryConfig:
    if retry_config is not None:
        return retry_config
    return JointFMRetryConfig(
        max_attempts=config.transport.retry.max_attempts,
        backoff_seconds=config.transport.retry.backoff_seconds,
        status_codes=config.transport.retry.status_codes,
        allowed_methods=config.transport.retryable_methods,
    )


def _resolve_response_body_excerpt_characters(
    response_body_excerpt_characters: int | None,
    config: JointFMConfig,
) -> int:
    if response_body_excerpt_characters is not None:
        return response_body_excerpt_characters
    return config.transport.response_body_excerpt_characters


def _resolve_datarobot_request_id_headers(
    datarobot_request_id_headers: Sequence[str] | None,
    config: JointFMConfig,
) -> Sequence[str]:
    if datarobot_request_id_headers is not None:
        return datarobot_request_id_headers
    return config.transport.datarobot_request_id_headers


def _is_history_row_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _forecast_response_from_payload(
    response_payload: Mapping[str, Any],
    request_payload: Mapping[str, Any],
) -> ForecastResponse:
    try:
        return ForecastResponse.from_payload(
            response_payload,
            request_payload=request_payload,
        )
    except JointFMServiceError:
        raise
    except ValueError as error:
        raise JointFMServiceError(
            f"JointFM forecast response violated the V1 contract: {error}"
        ) from error


def _sample_batch_cap_from_error(
    error: JointFMHTTPStatusError,
    payload: Mapping[str, Any],
) -> int | None:
    requested_samples = _requested_sample_count(payload)
    if requested_samples is None:
        return None

    cap_details = _sample_cap_details_from_error(error)
    if cap_details is None:
        return None
    reported_requested_samples, sample_cap = cap_details
    if reported_requested_samples != requested_samples:
        return None
    if requested_samples <= sample_cap:
        return None
    return sample_cap


def _known_sample_batch_cap(
    payload: Mapping[str, Any],
    sample_cap: int | None,
) -> int | None:
    if sample_cap is None:
        return None
    requested_samples = _requested_sample_count(payload)
    if requested_samples is None:
        return None
    if requested_samples <= sample_cap:
        return None
    return sample_cap


def _requested_sample_count(payload: Mapping[str, Any]) -> int | None:
    if payload.get("return_mode") != "samples":
        return None
    requested_samples = payload.get("n_samples")
    if isinstance(requested_samples, bool) or not isinstance(requested_samples, int):
        return None
    return requested_samples


def _sample_cap_details_from_error(
    error: JointFMHTTPStatusError,
) -> tuple[int, int] | None:
    for jointfm_error in error.jointfm_errors:
        message = jointfm_error.get("message")
        if isinstance(message, str):
            cap_details = _sample_cap_details_from_text(message)
            if cap_details is not None:
                return cap_details
    return _sample_cap_details_from_text(error.response_body_excerpt)


def _sample_cap_details_from_text(text: str) -> tuple[int, int] | None:
    match = _SAMPLE_CAP_ERROR_PATTERN.search(text)
    if match is None:
        return None
    requested_samples = int(match.group("requested"))
    sample_cap = int(match.group("cap"))
    if sample_cap < 1:
        return None
    return requested_samples, sample_cap


def _set_batch_seed(batch_payload: dict[str, Any], batch_index: int) -> None:
    seed = batch_payload.get("seed")
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        return
    batch_payload["seed"] = seed + batch_index


def _merge_sample_forecast_results(
    batch_results: Sequence[SampleForecastResult],
    request_payload: Mapping[str, Any],
) -> SampleForecastResult:
    if not batch_results:
        raise ValueError("sample batching produced no responses")

    first_result = batch_results[0]
    merged_samples = tuple(
        sample_values
        for batch_result in batch_results
        for sample_values in batch_result.samples
    )
    _validate_sample_batch_results(batch_results, first_result)
    requested_samples = cast(int, request_payload["n_samples"])
    if len(merged_samples) != requested_samples:
        raise ValueError(
            "sample batching produced the wrong sample count: "
            f"expected {requested_samples}, got {len(merged_samples)}"
        )

    seed = request_payload.get("seed")
    diagnostics_seed = (
        seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    )
    return SampleForecastResult(
        schema_version=first_result.schema_version,
        image_version=first_result.image_version,
        model_version=first_result.model_version,
        checkpoint_version=first_result.checkpoint_version,
        head=first_result.head,
        query_mode=first_result.query_mode,
        return_mode=first_result.return_mode,
        query_times=first_result.query_times,
        requested_columns=first_result.requested_columns,
        diagnostics=ForecastDiagnostics(
            history_rows=first_result.diagnostics.history_rows,
            horizon_count=first_result.diagnostics.horizon_count,
            seed=diagnostics_seed,
        ),
        errors=(),
        samples=merged_samples,
    )


def _validate_sample_batch_results(
    batch_results: Sequence[SampleForecastResult],
    first_result: SampleForecastResult,
) -> None:
    for batch_result in batch_results[1:]:
        if batch_result.schema_version != first_result.schema_version:
            raise ValueError("sample batch schema_version mismatch")
        if batch_result.image_version != first_result.image_version:
            raise ValueError("sample batch image_version mismatch")
        if batch_result.model_version != first_result.model_version:
            raise ValueError("sample batch model_version mismatch")
        if batch_result.checkpoint_version != first_result.checkpoint_version:
            raise ValueError("sample batch checkpoint_version mismatch")
        if batch_result.head != first_result.head:
            raise ValueError("sample batch head mismatch")
        if batch_result.query_mode != first_result.query_mode:
            raise ValueError("sample batch query_mode mismatch")
        if batch_result.return_mode != first_result.return_mode:
            raise ValueError("sample batch return_mode mismatch")
        if batch_result.query_times != first_result.query_times:
            raise ValueError("sample batch query_times mismatch")
        if batch_result.requested_columns != first_result.requested_columns:
            raise ValueError("sample batch requested_columns mismatch")
        if (
            batch_result.diagnostics.history_rows
            != first_result.diagnostics.history_rows
        ):
            raise ValueError("sample batch history_rows mismatch")
        if (
            batch_result.diagnostics.horizon_count
            != first_result.diagnostics.horizon_count
        ):
            raise ValueError("sample batch horizon_count mismatch")
