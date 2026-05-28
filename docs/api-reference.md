# JointFM Python SDK API Reference

This reference covers the supported public Python surface exported by `jointfm_client`. Names that start with an underscore are implementation details and are not part of the package contract.

## Client Entry Point

| Name | Purpose |
| --- | --- |
| `JointFMClient` | Synchronous client for hosted or local JointFM endpoints. Use `from_env()` for `.env` and `config.yaml` backed hosted settings, `health()` for `/healthz`, `predict(payload)` for low-level JSON prediction, `forecast(...)` for validated tabular forecasts, and the `forecast_mean(...)`, `forecast_samples(...)`, and `forecast_quantiles(...)` convenience methods for typed forecast results. |

`JointFMClient.from_env()` loads `config.yaml`, optional `.env` values, and process environment variables. `JointFMClient.health(cache=True)` caches health metadata only when requested. `JointFMClient.predict(payload)` requires `payload["model_version"]`; high-level forecast helpers resolve the configured model version when the caller does not pass one explicitly. When `forecast_samples(...)` requests more samples than the service cap allows, the client discovers the cap from the structured service error, resubmits capped prediction batches, and returns one merged `SampleForecastResult`.

## Contract Classes

| Name | Purpose |
| --- | --- |
| `ColumnSpec` | Describes one modeled request column. Fields are `name`, `modality`, `role`, `nullable`, `vocabulary_size`, `level_count`, `mapping`, `lower_bound`, `upper_bound`, `time_value_kind`, `time_value_scale_seconds`, `time_value_use_local_normalized_time`, `time_value_calendar_id`, and `time_value_timezone`. |
| `DataFrameSchema` | Describes tabular history layout. Fields are `columns`, `time_index_mode`, `time_column`, `time_scale_seconds`, `use_local_normalized_time`, `calendar_id`, and `timezone`. |
| `ForecastRequestMetadata` | Holds `schema_version`, `model_version`, `query_mode`, and `return_mode` for one forecast request. |
| `ForecastRequest` | Validated request object that combines metadata, schema, history rows, query times, requested columns, sample or quantile controls, and `seed`, then emits a JSON-compatible payload with `to_payload()`. |
| `HealthMetadata` | Typed `/healthz` payload with service status, schema and model versions, checkpoint metadata, device, head, advertised modes, and time-index encoding. |
| `StructuredError` | One structured JointFM service error with `code`, `message`, and optional `field`. |
| `ForecastDiagnostics` | Response diagnostics containing `history_rows`, `horizon_count`, and optional `seed`. |
| `QuantileForecast` | One quantile surface with `quantile` and `values`; `to_numpy()` returns axis order `(horizon, column)`. |
| `ForecastOutputs` | Legacy nested view of parsed output arrays, including `query_times`, `requested_columns`, and exactly one of `mean`, `samples`, or `quantiles`. |
| `ForecastResponse` | Shared base for parsed forecast results. It preserves schema, image, model, checkpoint, head, mode, query-time, requested-column, diagnostic, and error metadata. |
| `MeanForecastResult` | Parsed mean forecast. `to_numpy()` returns `(horizon, column)` and pandas helpers return tidy or wide frames. |
| `SampleForecastResult` | Parsed sample forecasts. `to_numpy()` returns `(sample, horizon, column)` and pandas helpers return tidy or wide frames. |
| `QuantileForecastResult` | Parsed quantile forecasts. `to_numpy()` returns `(quantile, horizon, column)`, `quantile_levels` exposes the ordered levels, and pandas helpers return tidy or wide frames. |

## Configuration Classes

| Name | Purpose |
| --- | --- |
| `JointFMSettings` | Validated hosted or local service settings: optional normalized DataRobot endpoint, optional secret token, health and prediction URLs, service selector, schema pin, model pin, and optional selector details. The API token is excluded from `repr`. |
| `JointFMConfig` | Top-level structured configuration loaded from defaults, YAML, and explicit overrides. |
| `PathConfig` | Default local file names for `config.yaml`, `config.sample.yaml`, and `.env`. |
| `EnvironmentVariableConfig` | Environment variable names consumed by settings loading. |
| `HostedDeploymentConfig` | Optional service target values layered below `.env` and process environment variables. |
| `TransportConfig` | Default HTTP transport settings: timeouts, retry policy, response excerpt length, retryable methods, DataRobot request-id headers, and user-agent header. |
| `TimeoutConfig` | YAML-backed connect and read timeout values. |
| `RetryConfig` | YAML-backed retry attempts, backoff, and retryable HTTP status codes. |
| `ForecastConfig` | Default forecast controls used by client and adapter helpers. |
| `ForecastCsvConfig` | Defaults used by the `forecast-csv` CLI command. |
| `CLIConfig` | Command line defaults, including `.env` path and CSV forecast defaults. |

## Transport Classes

| Name | Purpose |
| --- | --- |
| `JSONTransport` | Protocol for test transports or alternate JSON transports. It requires `get_json(url)` and `post_json(url, payload)`. |
| `JointFMHTTPTransport` | `requests.Session` backed JSON transport with configured headers, user agent, timeouts, retries, response decoding, request-id capture, and typed SDK exceptions. |
| `JointFMTimeoutConfig` | Runtime connect and read timeout settings consumed by `requests`. |
| `JointFMRetryConfig` | Runtime retry policy converted to the `urllib3` retry object used by `requests` adapters. |

## Exceptions

All SDK-specific exceptions inherit from `JointFMError`.

| Name | Raised When |
| --- | --- |
| `JointFMConfigurationError` | Local settings, credentials, URLs, deployment selection, or request configuration are invalid. |
| `JointFMTransportError` | Base class for transport failures. |
| `JointFMRequestEncodingError` | A request payload cannot be JSON encoded. |
| `JointFMRequestError` | A network request fails before a usable response is received. |
| `JointFMResponseError` | Base class for malformed or unsuccessful service responses. It preserves `status_code`, `response_body_excerpt`, optional `datarobot_request_id`, and structured JointFM errors. |
| `JointFMResponseDecodeError` | A service response is empty, non-JSON, or not a JSON object. |
| `JointFMHTTPStatusError` | The service returns an HTTP error status. |
| `JointFMServiceError` | A response body contains non-empty JointFM `errors`, including the case where HTTP status unexpectedly succeeded. |
| `JointFMCompatibilityError` | Base class for fail-fast service compatibility failures. |
| `UnsupportedSchemaVersionError` | The service or response advertises a schema version other than `v1`. |
| `UnsupportedModelVersionError` | The service or response model version differs from the configured or requested version. |
| `UnsupportedServiceContractError` | `/healthz` advertises mode capabilities outside the SDK's V1 contract. |

## Public Functions

| Name | Purpose |
| --- | --- |
| `load_configuration(config_path=..., overrides=...)` | Load structured SDK configuration from defaults, optional YAML, and explicit overrides. |
| `load_settings(env=..., dotenv_path=..., config_path=..., config=...)` | Load and validate hosted DataRobot or direct local REST settings from configuration, `.env`, and environment values. |
| `normalize_datarobot_endpoint(value)` | Validate and normalize a DataRobot API v2 endpoint ending in `/api/v2`. |
| `validate_datarobot_api_token(value)` | Validate a non-empty, whitespace-free token without exposing it. |
| `normalize_deployment_id(value)` | Validate a deployment ID as one non-empty path segment. |
| `normalize_hosted_deployment_url(value)` | Validate a hosted deployment URL ending in `/deployments/{deployment_id}`. |
| `normalize_hosted_predict_url(value)` | Validate a hosted prediction URL ending in `/predictionsUnstructured`. |
| `normalize_local_service_base_url(value)` | Validate a direct local JointFM service base URL. |
| `build_hosted_deployment_url(datarobot_endpoint, deployment_id)` | Build `.../deployments/{deployment_id}` from the DataRobot endpoint. |
| `build_hosted_predict_url(datarobot_endpoint, deployment_id)` | Build the hosted `predictionsUnstructured` URL from the DataRobot endpoint and deployment ID. |
| `build_hosted_health_url(datarobot_endpoint, deployment_id)` | Build the hosted health URL from the DataRobot endpoint and deployment ID. |
| `build_hosted_predict_url_from_deployment_url(deployment_url)` | Append `/predictionsUnstructured` to a normalized hosted deployment URL. |
| `build_hosted_health_url_from_deployment_url(deployment_url)` | Append `/healthz` to a normalized hosted deployment URL. |
| `deployment_id_from_hosted_deployment_url(deployment_url)` | Extract a deployment ID from a hosted deployment URL. |
| `deployment_url_from_hosted_predict_url(predict_url)` | Return the owning hosted deployment URL for a hosted prediction URL. |
| `build_local_health_url(service_base_url)` | Build a direct local service `/healthz` URL. |
| `build_local_predict_url(service_base_url)` | Build a direct local service `/predict` URL. |
| `build_datarobot_prediction_headers(api_token)` | Build hosted prediction headers: bearer authorization, broad accept header, and JSON content type. |
| `build_forecast_payload(...)` | Build a validated JSON-compatible V1 forecast payload from explicit schema, history rows, query times, and return-mode controls. |
| `validate_service_metadata(metadata, expected_model_version=None)` | Validate `/healthz` metadata against schema `v1`, the expected model when supplied, and advertised V1 mode capabilities. |
| `infer_column_specs_from_dataframe(frame, ...)` | Infer ordered `ColumnSpec` objects from a pandas `DataFrame` and explicit role, modality, mapping, nullability, time-value, and bounds hints. |
| `dataframe_to_history_rows(frame, schema)` | Convert a pandas `DataFrame` into server-compatible `history_rows`. |
| `arrays_to_history_rows(values, columns=..., ...)` | Convert a two-dimensional NumPy-like array plus column metadata into `history_rows`. |
| `build_forecast_payload_from_dataframe(frame, ...)` | Build a validated forecast payload from a pandas `DataFrame`. |
| `build_forecast_payload_from_arrays(values, ...)` | Build a validated forecast payload from a two-dimensional NumPy-like array. |
| `build_datetime_query_times(history_times, periods=..., frequency=...)` | Build future absolute-datetime query times from ordered history times. |
| `build_ordinal_query_times(history_times, periods=..., step=1)` | Build future ordinal query times from ordered history times. |
| `build_continuous_query_times(history_times, periods=..., step=...)` | Build future continuous-float query times from ordered history times. |
| `validate_forecast_horizon(history_times, query_times, time_index_mode=...)` | Validate that query times are future, increasing, and encoded for the selected time-index mode. |
| `resolve_notebook_project_root(start_dir=None)` | Resolve the nearest src-layout Python project root for notebooks started inside a repository tree. |
| `bootstrap_notebook(add_src_root=False)` | Switch notebook working directory to the project root and optionally prepend the local `src` directory. |

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATAROBOT_ENDPOINT` | Hosted calls | HTTPS DataRobot API v2 endpoint, normalized without a trailing slash and required to end in `/api/v2`. |
| `DATAROBOT_API_TOKEN` | Hosted calls | Non-empty, whitespace-free API token used in the hosted bearer authorization header. |
| `JOINTFM_SCHEMA_VERSION` | Hosted calls | Request schema pin. The SDK supports only `v1`. |
| `JOINTFM_MODEL_VERSION` | Hosted calls | Exact JointFM deployment model version expected from `/healthz` and prediction responses. |
| `JOINTFM_DEPLOYMENT_ID` | One selector | Deployment ID used to build hosted health and prediction URLs. |
| `JOINTFM_DEPLOYMENT_URL` | One selector | Hosted deployment URL; the SDK derives `/healthz` and `/predictionsUnstructured`. |
| `JOINTFM_PREDICT_URL` | One selector | Full hosted prediction URL ending in `/predictionsUnstructured`; the SDK derives the owning deployment URL. |
| `JOINTFM_DEPLOYMENT_TARGET` | One selector with outputs path | Key in a saved Pulumi outputs JSON file. |
| `JOINTFM_PULUMI_OUTPUTS_PATH` | With target selector | JSON file containing target outputs with exactly one of `deployment_id`, `deployment_url`, or `predict_url`. |
| `JOINTFM_LOCAL_BASE_URL` | One selector | Direct local JointFM REST service base URL. The SDK calls `GET /healthz` and `POST /predict` without DataRobot authorization. |
| `DATAROBOT_DEPLOYMENT_ID` | Optional live tests | Hosted deployment ID used only by the optional live smoke test so normal CI does not call DataRobot accidentally. |

Set exactly one selector among `JOINTFM_DEPLOYMENT_ID`, `JOINTFM_DEPLOYMENT_URL`, `JOINTFM_PREDICT_URL`, `JOINTFM_DEPLOYMENT_TARGET`, and `JOINTFM_LOCAL_BASE_URL`.

## V1 Payload Fields

### Forecast Request

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | Yes | Must be `"v1"`. |
| `model_version` | Yes | Exact deployed model version expected by the caller. |
| `query_mode` | Yes | Must be `"forecast"`. |
| `return_mode` | Yes | One of `"mean"`, `"samples"`, or `"quantiles"`. |
| `time_index_mode` | Yes | One of `"ordinal"`, `"continuous_float"`, or `"absolute_datetime"`. |
| `columns` | Yes | Non-empty array of column descriptors for modeled columns. |
| `history_rows` | Yes | Non-empty array of history row objects in the declared schema. |
| `query_times` | Yes | Non-empty future forecast horizon values. Absolute datetimes are encoded timezone-stably. |
| `time_column` | For absolute datetime, optional otherwise | Name of the history time column. It must not duplicate a modeled column name. |
| `requested_columns` | Optional | Output column names or integer indices. Duplicates are rejected. Defaults to all modeled columns. |
| `n_samples` | Samples and quantiles controls | Positive sample count when sampling controls are needed. Oversized sample forecasts are batched automatically after the service reports its cap. |
| `quantiles` | Quantiles mode | Quantile levels in `(0, 1)`, required for `return_mode="quantiles"`. |
| `seed` | Optional | Integer random seed for reproducible stochastic outputs. |
| `time_scale_seconds` | Optional | Positive scale for continuous time indexes. |
| `use_local_normalized_time` | Optional | Whether the service should use local normalized time features. |
| `calendar_id` | Optional | Calendar identifier, defaulting to `pandas-default`. |
| `timezone` | Optional | Time zone for absolute datetime handling. |

### Column Descriptor

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Unique modeled column name. |
| `modality` | Yes | One of `numeric`, `categorical`, `ordinal`, `binary`, `count`, or `time_value`. |
| `role` | Optional | One of `target`, `known_dynamic`, `past_dynamic`, `static`, or `feature`; omitted values default to `feature`. |
| `nullable` | Optional | Whether null values are accepted for the column. |
| `vocabulary_size` | Categorical metadata | Positive category vocabulary size. |
| `level_count` | Ordinal metadata | Positive number of ordered levels. |
| `mapping` | Categorical or ordinal metadata | Mapping from raw values to integer category or level IDs. |
| `lower_bound` | Numeric metadata | Optional finite lower bound. |
| `upper_bound` | Numeric metadata | Optional finite upper bound; must not be less than `lower_bound`. |
| `time_value_kind` | Time-value columns | One of `continuous_float` or `absolute_datetime` when `modality="time_value"`. |
| `time_value_scale_seconds` | Time-value columns | Positive scale for time-valued numeric data. |
| `time_value_use_local_normalized_time` | Time-value columns | Whether local normalized time features are used. |
| `time_value_calendar_id` | Time-value columns | Calendar identifier, defaulting to `pandas-default`. |
| `time_value_timezone` | Time-value columns | Time zone for absolute time-valued columns. |

### Forecast Response

| Field | Description |
| --- | --- |
| `schema_version` | Response schema, expected to be `"v1"`. |
| `image_version` | Service image version that produced the response. |
| `model_version` | Model version that produced the response. |
| `checkpoint_version` | Checkpoint version that produced the response. |
| `head` | Forecast head used by the service. |
| `query_mode` | Response query mode, expected to be `"forecast"`. |
| `return_mode` | Response return mode matching the request. |
| `outputs.query_times` | Forecast horizon values preserved from the request. |
| `outputs.requested_columns` | Output columns in response order. |
| `outputs.mean` | Mean values with axis order `(horizon, column)` when `return_mode="mean"`. |
| `outputs.samples` | Sample values with axis order `(sample, horizon, column)` when `return_mode="samples"`. |
| `outputs.quantiles` | Quantile surfaces with axis order `(quantile, horizon, column)` when `return_mode="quantiles"`. |
| `diagnostics.history_rows` | Number of history rows processed. |
| `diagnostics.horizon_count` | Number of forecast horizon steps returned. |
| `diagnostics.seed` | Optional seed used by the service. |
| `errors` | Structured service errors. Non-empty arrays raise typed SDK exceptions. |

### Health Metadata

| Field | Description |
| --- | --- |
| `status` | Service status string. |
| `schema_version` | Advertised schema version. The SDK requires `v1`. |
| `image_version` | Running service image version. |
| `model_version` | Running model version. |
| `checkpoint_version` | Loaded checkpoint version. |
| `checkpoint_path` | Loaded checkpoint path reported by the service. |
| `device` | Device used by inference. |
| `head` | Active forecast head. |
| `supported_query_modes` | Must match the SDK V1 query modes. |
| `supported_return_modes` | Must match the SDK V1 return modes. |
| `supported_time_index_modes` | Must match the SDK V1 time-index modes. |
| `time_index_encoding` | Time-index encoding advertised by the service. |

## Docstring Enforcement

Ruff is configured in `pyproject.toml` to require module, class, method, and function docstrings with `D100`, `D101`, `D102`, and `D103`. The repository pre-commit configuration runs `uv run ruff check .`, so `task pre-commit` fails if public documentation strings are missing.