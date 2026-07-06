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

`JointFMClient.from_env()` and `load_settings()` resolve `JOINTFM_SCHEMA_VERSION` and exactly one service selector from that layered configuration. `JOINTFM_MODEL_VERSION` is optional: when unset the SDK discovers the model version from `/healthz` on first use, and when set the SDK validates it against `/healthz` as a drift-detection guard. Hosted selectors also require `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`; the direct local selector does not use DataRobot credentials. Missing credentials, missing schema version, malformed credentials, unsupported schema versions, missing selectors, and multiple selectors raise `JointFMConfigurationError`.

`DATAROBOT_ENDPOINT` must be a normalized HTTPS DataRobot API v2 URL ending in `/api/v2`; the SDK stores it without a trailing slash. `DATAROBOT_API_TOKEN` must be non-empty and whitespace-free. The token is excluded from `JointFMSettings` repr output.

Required `.env` entries for hosted SDK calls are `DATAROBOT_ENDPOINT`, `DATAROBOT_API_TOKEN`, `JOINTFM_SCHEMA_VERSION`, and exactly one hosted selector from the list below. Required `.env` entries for local REST calls are `JOINTFM_LOCAL_BASE_URL` and `JOINTFM_SCHEMA_VERSION`. `JOINTFM_MODEL_VERSION` may be set in `.env` to pin a specific deployment artifact (the SDK then hard-errors on mismatch with `/healthz`); leave it unset to let the SDK use whatever version the deployment currently advertises. `.env` is the right place for these pins when using `from_env()` because they describe the selected JointFM service rather than a package-wide default. They are not secrets, and callers can still override them with process environment variables. Optional live DataRobot smoke tests additionally read `DATAROBOT_DEPLOYMENT_ID` from `.env` and use it as the hosted deployment ID for the `deployments/{deployment_id}/predictionsUnstructured` route.

Example deployment configuration:

```yaml
deployment:
	datarobot_endpoint: https://app.datarobot.com/api/v2
	datarobot_api_token: <token>
	schema_version: v1
	deployment_id: <deployment-id>
	# Optional model-version pin; the SDK discovers it from /healthz when unset:
	# model_version: jointfm-inference:0.2.0+ckpt.fin-2026-05-22
transport:
	timeout:
		connect_seconds: 5.0
		read_seconds: 60.0
	retry:
		max_attempts: 3
		backoff_seconds: 1
```

Equivalent `.env` deployment configuration:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v1
JOINTFM_DEPLOYMENT_ID=<deployment-id>
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin-2026-05-22
```

Equivalent local REST configuration for a service started from the `joint` repository with `task service:start CONFIG=nvidia-studentt-m4cr2`:

```dotenv
JOINTFM_LOCAL_BASE_URL=http://127.0.0.1:8080
JOINTFM_SCHEMA_VERSION=v1
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin_i504_o63_f0_t10_h16l16_mam7_af_t3r1_cnn_k3l4_hpst_h16l2_studentt_m4cr2df8skew
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
- `model_version`: exact model version advertised by `/healthz` or otherwise selected by the caller. Optional for `from_env()` clients: when `JOINTFM_MODEL_VERSION` is unset the SDK reads it from `/healthz` on first use; when set it acts as a drift-detection pin
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

Create `.env` from `.env.sample` or set the same values in your shell. A hosted forecast needs the DataRobot API v2 endpoint, token, schema pin, and exactly one deployment selector:

```dotenv
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<token>
JOINTFM_SCHEMA_VERSION=v1
JOINTFM_DEPLOYMENT_ID=<deployment-id>
# Optional drift-detection pin; the SDK discovers the model version from /healthz when unset:
# JOINTFM_MODEL_VERSION=jointfm-inference:0.2.0+ckpt.fin-2026-05-22
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

- `task lint`: run Ruff lint checks (read-only)
- `task format`: run Ruff formatter (rewrites files in place)
- `task license-check`: verify every Python source file has the required copyright/SPDX header
- `task typecheck`: run `ty` static type checks
- `task test`: run unit tests
- `task coverage`: run tests with coverage enforcement above 90%
- `task build`: build the source distribution and wheel, then validate artifact metadata and contents
- `task check`: run the static code quality gate (typos, lint, format check, type checks)
- `task release:dry`: preview the next SemVer bump without changing any files
- `task release`: cut a SemVer release with Commitizen (writes `CHANGELOG.md`, bumps versions, creates tag)
- `task pre-commit`: run every configured pre-commit hook

Contributors do not need to add copyright or license headers to new Python files manually. The `insert-license` pre-commit hook stamps the standard SPDX header (`Copyright (c) 2026 DataRobot, Inc.` + `SPDX-License-Identifier: Apache-2.0`) into every `.py` file the first time you run `task pre-commit`. Verify the headers are present at any time with `task license-check`.

## Versioning & Commits

The package follows strict [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are cut with [Commitizen](https://commitizen-tools.github.io/commitizen/), driven by [Conventional Commits](https://www.conventionalcommits.org/), so the commit log is the source of truth for what a release contains.

### Commit message format

Every commit subject must follow:

```text
<type>(<optional scope>): <imperative summary>
```

Examples:

```text
feat(adapters): add quantile forecast helper
fix(transport): retry on idempotent 5xx responses
perf(adapters): cache schema validation across forecast batches
refactor(configuration): split URL resolution into helpers
docs(readme): document the release workflow
test(transport): cover retry on 502/503/504
build(deps): bump pydantic to 2.13.4
ci(pre-commit): pin commitizen to v3.31
chore(repo): move fixtures to tests/fixtures/v1
```

#### Commit types

The full set of accepted types comes from the `cz_conventional_commits` rule set referenced above. Pick the type that matches the **primary intent** of the commit; split commits that mix concerns. Only `feat`, `fix`, `refactor`, and `perf` (plus breaking-change markers) drive a SemVer bump — every other type is recorded in `git log` but does not, on its own, cause `cz bump` to cut a new version.

| Type | When to use | SemVer bump |
| --- | --- | --- |
| `feat` | A user-visible feature: new public API, new CLI flag, new helper exposed to callers. | **minor** |
| `fix` | A bug fix in shipped behavior — the symptom is observable to callers or operators. | **patch** |
| `perf` | A change that improves performance without changing observable behavior. | **patch** |
| `refactor` | An internal restructure that neither adds a feature nor fixes a bug (renames, moves, extractions, internal type changes). | **patch** |
| `docs` | Documentation-only changes (READMEs, prose, docstrings, comments, example payloads, API reference). | none |
| `test` | Adding, fixing, or restructuring tests, fixtures, or test helpers with no production-code change. | none |
| `build` | Build system, packaging, or dependency-pinning changes (`pyproject.toml`, lockfile, wheel build hooks). | none |
| `ci` | CI configuration changes (GitHub Actions workflows, `.pre-commit-config.yaml`, release hooks). | none |
| `chore` | Repository maintenance not covered above (tooling tweaks, repo-level renames, housekeeping, fixture moves). | none |
| `style` | Pure formatting or whitespace, no behavior change — rare here because Ruff format runs in pre-commit. | none |
| `revert` | Reverts a previous commit; the body should include `Refs: <sha>`. Re-add `!` or a `BREAKING CHANGE:` footer if the reverted commit was a breaking change. | none |
| any type with `!` or a `BREAKING CHANGE:` footer | A breaking change to public API, configuration schema, environment variables, or the V1 wire contract. | **major**¹ |

¹ This project sets `major_version_zero = true` in `pyproject.toml`, so breaking changes are downgraded to a **minor** bump while the SDK is on `0.x`. They will become major bumps once the SDK ships `1.0.0`.

A release window that contains only `none`-bump commits is not releasable on its own: `cz bump` exits without writing a new version. Either land a `feat`/`fix`/`refactor`/`perf` first, or force the bump explicitly with `task release -- --increment PATCH`.

#### Breaking changes

Declare a breaking change with `!` after the type, e.g. `feat(adapters)!: rename forecast() to predict()`, or — preferred when the change needs explanation — with an explicit footer:

```text
feat(adapters): rename forecast() to predict()

BREAKING CHANGE: forecast() is gone; callers must use predict().
```

The `commit-msg` pre-commit hook runs `cz check` and rejects malformed messages locally before the commit is created.

### First-time tag seed (one-off)

Commitizen bumps **from** the tag matching the current `version =` in `pyproject.toml`. A brand-new repo has no such tag, so the first `task release` or `task release:dry` will fail with a clear message. Seed it once:

```bash
git tag -a v0.0.1 -m 'Seed initial release tag for Commitizen'
git push --tags
```

After that, every future `task release` finds its base tag automatically.

### Cutting a release

```bash
task release:dry      # preview the next version + CHANGELOG entries
task release          # bump, write CHANGELOG.md, create the annotated tag
git push && git push --tags
```

`task release` first runs `task release:check` (clean tree, on `main`, in sync with `origin/main`), then calls `cz bump` which:

- reads commits since the last `v*` tag,
- picks the SemVer bump from the types it sees,
- updates `CHANGELOG.md`,
- bumps `version =` in `pyproject.toml`, `__version__` in `src/jointfm_client/__init__.py`, and the "Current SDK package version" line in this README,
- commits the bump and creates the annotated tag.

Pushing is left manual so you can inspect the bump first. Override the inferred bump level only when needed: `task release -- --increment minor`.

Pushing the `v*` tag triggers the [`Publish to PyPI`](.github/workflows/publish.yml) workflow, which rebuilds and validates the distribution with `task build` and uploads it to PyPI via [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish) — keeping the git tag, the wheel filename, `jointfm_client.__version__`, and the PyPI version all in lockstep.

The workflow authenticates with PyPI through [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no API-token secret is stored in the repository. Before the first release, configure it once on PyPI:

- register a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for the `jointfm-client` project pointing at owner `datarobot`, repository `joint-client-python`, workflow `publish.yml`, and environment `pypi`;
- create a GitHub Actions environment named `pypi` on the repository (optionally gated with required reviewers) so the publish job can run.

After pulling these changes for the first time, run `uv run pre-commit install` (or `task setup`) once so the new `commit-msg` hook is registered with git.

### Hand-editing the version (don't)

The `version =` line in [`pyproject.toml`](pyproject.toml), `__version__` in [`src/jointfm_client/__init__.py`](src/jointfm_client/__init__.py), and the "Current SDK package version" line in this README are **owned by `cz bump`** — treat them the way you'd treat a lockfile. Each release rewrites all of them atomically.

If you hand-edit them, the next `task release` will catch you:

- if you bumped only some lines, `cz bump` aborts with "Configured files cannot be updated, check consistency";
- if you bumped them all but never created the matching `vX.Y.Z` tag, `task release:require-tag` refuses to run and tells you to seed the tag.

If you genuinely need to set a specific version outside the normal release flow (bootstrap, recovery from a botched state), use the escape hatch:

```bash
uv run cz bump --files-only X.Y.Z
```

That atomically rewrites every `version_files` line to `X.Y.Z` without committing or tagging. After it returns, commit and tag manually.
