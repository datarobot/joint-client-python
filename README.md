# JointFM Python SDK

`jointfm-client` is the Python SDK package for callers of the JointFM REST API. The import namespace is `jointfm_client`, and the first supported Python version is Python 3.13.

The SDK targets the DataRobot-hosted unstructured prediction route and the same direct local service contract used by the JointFM inference container. Public code for the contract lives in `jointfm_client.contract`; the README mirrors it for package users.

## Package Contract

- Distribution package: `jointfm-client`
- Import namespace: `jointfm_client`
- Supported Python: `>=3.13`
- Current SDK package version: `0.0.1`
- Current JointFM service schema: `schema_version="v1"`

The public API shape is a synchronous low-level `JointFMClient` with `health()` and `predict(payload)` methods plus high-level `forecast(...)`, `forecast_mean(...)`, `forecast_samples(...)`, and `forecast_quantiles(...)` helpers. The SDK is not a proxy service; callers use it as a local Python library that talks to the hosted or local JointFM endpoint.

SDK package versions are standard Python distribution versions: `[project].version` in `pyproject.toml` and `jointfm_client.__version__` describe the released client library. JointFM `schema_version`, `image_version`, `model_version`, and `checkpoint_version` are service compatibility identifiers carried in configuration, health metadata, requests, and responses. They are not SDK package versions, and changing a deployment pin does not by itself require changing the SDK package version.

See [docs/api-reference.md](docs/api-reference.md) for the checked-in API reference covering public classes, functions, exceptions, environment variables, and V1 payload fields.

## Service Contract

The DataRobot-hosted prediction URL is built from the DataRobot API v2 endpoint and deployment ID:

```python
from urllib.parse import urljoin

service_base_url = DATAROBOT_ENDPOINT.rstrip("/") + "/"
predict_url = urljoin(
		service_base_url,
		f"deployments/{deployment_id}/predictionsUnstructured",
)
```

The direct local service exposes `GET /healthz` and `POST /predict`.

## Configuration And Authentication

Structured SDK defaults live in `jointfm_client.configuration.JointFMConfig` and are mirrored in the checked-in `config.sample.yaml`. Copy `config.sample.yaml` to `config.yaml` and change only the fields needed for your deployment or transport defaults. `JointFMClient.from_env()` and `load_settings()` read `config.yaml` by default, then layer `.env` values over it, then layer process environment variables or the supplied `env` mapping over both. Explicit Python arguments such as `timeout=` and `retry_config=` still override YAML transport defaults.

`JointFMClient.from_env()` and `load_settings()` resolve `JOINTFM_SCHEMA_VERSION`, `JOINTFM_MODEL_VERSION`, and exactly one service selector from that layered configuration. Hosted selectors also require `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`; the direct local selector does not use DataRobot credentials. Missing credentials, missing version pins, malformed credentials, unsupported schema versions, missing selectors, and multiple selectors raise `JointFMConfigurationError`.

`DATAROBOT_ENDPOINT` must be a normalized HTTPS DataRobot API v2 URL ending in `/api/v2`; the SDK stores it without a trailing slash. `DATAROBOT_API_TOKEN` must be non-empty and whitespace-free. The token is excluded from `JointFMSettings` repr output.

Required `.env` entries for hosted SDK calls are `DATAROBOT_ENDPOINT`, `DATAROBOT_API_TOKEN`, `JOINTFM_SCHEMA_VERSION`, `JOINTFM_MODEL_VERSION`, and exactly one hosted selector from the list below. Required `.env` entries for local REST calls are `JOINTFM_LOCAL_BASE_URL`, `JOINTFM_SCHEMA_VERSION`, and `JOINTFM_MODEL_VERSION`. `.env` is the right place for these pins when using `from_env()` because they describe the selected JointFM service rather than a package-wide default. They are not secrets, and callers can still override them with process environment variables. Optional live DataRobot smoke tests additionally read `DATAROBOT_DEPLOYMENT_ID` from `.env` and use it as the hosted deployment ID for the `deployments/{deployment_id}/predictionsUnstructured` route.

Example deployment configuration:

```yaml
deployment:
	datarobot_endpoint: https://app.datarobot.com/api/v2
	datarobot_api_token: <token>
	schema_version: v1
	model_version: jointfm-inference:0.2.0+ckpt.fin-2026-05-22
	deployment_id: <deployment-id>
transport:
	timeout:
		connect_seconds: 5.0
		read_seconds: 60.0
	retry:
		max_attempts: 3
		backoff_seconds: 0.25
```

Equivalent `.env` deployment configuration:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v1
JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin-2026-05-22
JOINTFM_DEPLOYMENT_ID=<deployment-id>
```

Equivalent local REST configuration for a service started from the `joint` repository with `task service:start CONFIG=nvidia-studentt-m4cr2`:

```dotenv
JOINTFM_LOCAL_BASE_URL=http://127.0.0.1:8080
JOINTFM_SCHEMA_VERSION=v1
JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin_i504_o63_f0_t10_h16l16_mam7_af_t3r1_cnn_k3l4_hpst_h16l2_studentt_m4cr2df8skew
```

Choose exactly one service selector:

- `JOINTFM_DEPLOYMENT_ID`: builds `DATAROBOT_ENDPOINT.rstrip("/") + "/"` plus `deployments/{deployment_id}/predictionsUnstructured`
- `JOINTFM_DEPLOYMENT_URL`: appends `/predictionsUnstructured` to a hosted deployment URL
- `JOINTFM_PREDICT_URL`: uses a full hosted prediction URL ending in `/predictionsUnstructured`
- `JOINTFM_DEPLOYMENT_TARGET` with `JOINTFM_PULUMI_OUTPUTS_PATH`: resolves a named target from saved Pulumi outputs JSON, preferring `deployment_id`, then `deployment_url`, then `predict_url`
- `JOINTFM_LOCAL_BASE_URL`: builds direct local `GET /healthz` and `POST /predict` URLs without DataRobot authentication

Pulumi deployment discovery is explicit and file-backed. Export stack outputs to a JSON object keyed by target name, then set `JOINTFM_DEPLOYMENT_TARGET` to the key and `JOINTFM_PULUMI_OUTPUTS_PATH` to that JSON file:

```json
{
	"fin-studentt": {
		"deployment_id": "<deployment-id>"
	}
}
```

For each target, the SDK accepts exactly one of these output fields: `deployment_id`, `deployment_url`, or `predict_url`. `deployment_id` is preferred because it lets the SDK build both the hosted health URL and the `deployments/{deployment_id}/predictionsUnstructured` URL from the configured `DATAROBOT_ENDPOINT`. `deployment_url` is normalized and extended with `/healthz` and `/predictionsUnstructured`. `predict_url` is accepted when the full hosted prediction URL has already been discovered; the SDK derives the owning deployment URL from it for health checks.

Hosted prediction calls use the same authorization scheme as the notebook helper:

```python
{
	"Authorization": f"Bearer {DATAROBOT_API_TOKEN}",
	"Accept": "*/*",
	"Content-Type": "application/json;charset=UTF-8",
}
```

The SDK still decodes hosted prediction responses as JSON; the broad `Accept` value avoids hosted unstructured prediction content negotiation failures before the deployment body is returned.

Direct local URL helpers are used by the local service selector: `build_local_health_url("http://localhost:8080")` returns `/healthz`, and `build_local_predict_url("http://localhost:8080")` returns `/predict`.

Hosted settings also derive `health_url` from the resolved deployment URL as `deployments/{deployment_id}/healthz`. `JointFMClient.health(cache=True)` stores typed `HealthMetadata` only when the caller asks for caching, and `JointFMClient.refresh_health()` fetches a fresh copy.

## CLI Workflows

The package installs a `jointfm-client` command. It reads `.env` by default, accepts `--dotenv <path>` for another file, and accepts `--no-dotenv` when the process environment should be the only source.

Validate credentials, resolve the deployment, call `/healthz`, and print non-secret service metadata:

```bash
uv run jointfm-client health
```

Submit one low-level JSON request file and write the JSON response file:

```bash
uv run jointfm-client predict request.json response.json
```

Forecast from CSV history and write tidy forecast rows as CSV:

```bash
uv run jointfm-client forecast-csv history.csv forecast.csv \
	--query-times 2,3,4 \
	--target-column target \
	--return-mode mean
```

`forecast-csv` supports `--time-index-mode ordinal|continuous_float|absolute_datetime`, `--time-column`, repeated `--target-column`, repeated `--requested-column`, `--return-mode mean|samples|quantiles`, `--n-samples`, `--quantiles`, and `--seed`. The output is the same tidy shape returned by the Python result helpers.

## Notebook Workflows

Checked-in example notebooks live under `notebooks/`. Every example starts with:

```python
from jointfm_client import bootstrap_notebook
bootstrap_notebook(add_src_root=True)
```

Run `task setup` first so VS Code can select the checked-in `Python (joint-client-python)` notebook kernel backed by this repository's `.venv`.

The bootstrap helper resolves the nearest src-layout Python project root, switches the working directory there, and prepends that project's local `src` tree during development. The examples cover hosted health checks, low-level JSON prediction, mean forecasts, sample forecasts, quantile forecasts, pandas/NumPy result conversion, and CSV forecast workflows. They use `.env.sample` placeholders and checked-in fixture payloads; no real tokens or deployment IDs are stored in notebooks.

The current V1 forecast request contract is:

- `schema_version`: exactly `"v1"`, configured as `JOINTFM_SCHEMA_VERSION` for `from_env()` clients
- `model_version`: exact model version advertised by `/healthz` or otherwise selected by the caller, configured as `JOINTFM_MODEL_VERSION` for `from_env()` clients
- `query_mode`: `"forecast"`
- `return_mode`: one of `"mean"`, `"samples"`, or `"quantiles"`
- `time_index_mode`: one of `"ordinal"`, `"continuous_float"`, or `"absolute_datetime"`
- `time_column`: required for `"absolute_datetime"`, and used for ordered ordinal or continuous histories when supplied
- `query_times`: non-empty future forecast times only
- `requested_columns`: optional column names or integer column indices, with duplicates rejected
- `n_samples`: positive sample count for sampled forecasts and quantile estimation. When `return_mode="samples"` exceeds a service-reported sample cap, `forecast_samples(...)` automatically resubmits capped prediction batches and returns one merged `SampleForecastResult`.

V1 column descriptors support the server fields `name`, `modality`, `role`, `nullable`, `vocabulary_size`, `level_count`, `mapping`, `lower_bound`, `upper_bound`, `time_value_kind`, `time_value_scale_seconds`, `time_value_use_local_normalized_time`, `time_value_calendar_id`, and `time_value_timezone`.

DataFrame helpers and the notebook examples are available through one optional extra that pulls in `pandas` and `yfinance` (the latter powers the Yahoo Finance download in `notebooks/forecast_trading.ipynb`):

```bash
uv add "jointfm-client[notebooks]"
```

Use `build_forecast_payload_from_dataframe(...)` when history is already in a pandas DataFrame. It can accept explicit `ColumnSpec` objects or infer basic numeric, categorical, ordinal, count, binary, and time-valued columns from the DataFrame plus role, mapping, nullable, and bounds hints. The helper emits `history_rows` in the same order as the service frame builder: `time_column` first when present, followed by the ordered modeled columns. `build_forecast_payload_from_arrays(...)` provides the same request path for two-dimensional NumPy-like arrays when callers already have array values and column metadata. `build_datetime_query_times(...)`, `build_ordinal_query_times(...)`, `build_continuous_query_times(...)`, and `validate_forecast_horizon(...)` perform local future-horizon validation before the SDK sends the request.

Successful forecast responses preserve `schema_version`, `image_version`, `model_version`, `checkpoint_version`, `head`, `query_mode`, `return_mode`, `outputs`, and `diagnostics`. Structured service errors use this shape:

```json
{
	"schema_version": "v1",
	"errors": [
		{
			"code": "VALIDATION_ERROR",
			"message": "request field explanation",
			"field": "request_field"
		}
	]
}
```

Known V1 error codes are `VALIDATION_ERROR`, `SCHEMA_VERSION_MISMATCH`, `MODEL_VERSION_MISMATCH`, `INPUT_SIZE_EXCEEDED`, and `INTERNAL_ERROR`.

## Compatibility Policy

The SDK supports only `schema_version="v1"`. `validate_service_metadata()` checks `/healthz` metadata and raises typed compatibility errors before prediction if the service advertises a different schema, an unexpected model version, or mode capabilities outside the recorded V1 contract.

Callers should pass an expected `model_version` when they already know which deployment artifact they intend to use. A mismatch is treated as a hard compatibility error rather than silently downgrading, guessing, or retrying another model.

High-level forecast helpers build the same validated request payloads as `build_forecast_payload(...)` and return `ForecastResponse`. Row-list inputs require an explicit `DataFrameSchema` or `ColumnSpec` sequence, while pandas DataFrame inputs can use the DataFrame adapter inference options. `predict(payload)` remains the low-level JSON method and requires the payload to include `model_version`.

## Quick Start

### Install `uv` and `task`

Use the same curl-based bootstrap flow on Ubuntu, macOS, and AWS Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13

sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

Ensure `~/.local/bin` is on your `PATH` in new shells.

### Set Up The Project

```bash
task setup
```

`task setup` creates or reuses `.venv` pinned to Python 3.13.3, synchronizes the project and development dependencies, installs the local Git pre-commit hooks, installs or verifies `typos`, and prints the shell activation hint for `.venv`.

Verify the installed package import:

```bash
uv run python -c "import jointfm_client; print(jointfm_client.__version__)"
```

### Configure A Deployment

Create `.env` from `.env.sample` or set the same values in your shell. A hosted forecast needs the DataRobot API v2 endpoint, token, schema and model pins, and exactly one deployment selector:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v1
JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin-2026-05-22
JOINTFM_DEPLOYMENT_ID=<deployment-id>
```

Check that the SDK can resolve the deployment and that the service metadata matches the configured schema and model pins:

```bash
uv run jointfm-client health
```

### Run The First Forecast

This minimal example uses row dictionaries plus explicit column metadata, so it does not require pandas:

```python
from jointfm_client import (
	ColumnSpec,
	DataFrameSchema,
	JointFMClient,
	build_ordinal_query_times,
)

history_rows = [
	{"t": 0, "sales": 10.0},
	{"t": 1, "sales": 12.0},
	{"t": 2, "sales": 13.5},
	{"t": 3, "sales": 15.0},
]
schema = DataFrameSchema(
	columns=(ColumnSpec(name="sales", modality="numeric", role="target"),),
	time_index_mode="ordinal",
	time_column="t",
)

client = JointFMClient.from_env()
result = client.forecast_mean(
	history_rows,
	schema=schema,
	query_times=build_ordinal_query_times([row["t"] for row in history_rows], periods=3),
	requested_columns=("sales",),
)

print(result.to_pandas_tidy())
```

Use `forecast_samples(...)` for sampled trajectories or `forecast_quantiles(...)` with `quantiles=(0.1, 0.5, 0.9)` for quantile surfaces. For pandas inputs, install the optional dataframe extra and pass a `DataFrame` to `forecast(...)` with `target_columns` and `requested_columns` set explicitly.

## Development Commands

- `task lint`: run Ruff checks
- `task typecheck`: run `ty` static type checks
- `task test`: run unit tests
- `task coverage`: run tests with coverage enforcement above 90%
- `task build`: build the source distribution and wheel, then validate artifact metadata and contents
- `task release-check`: run lint, type checks, coverage, and the validated build
- `task pre-commit`: run every configured pre-commit hook
