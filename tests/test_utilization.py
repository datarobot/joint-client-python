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

"""Tests for the maximum-utilization check of jointfm_client."""

from __future__ import annotations

from collections.abc import Mapping
import io
import threading
from typing import Any, cast

import pytest

from jointfm_client import (
    JointFMCapacityError,
    JointFMClient,
    JointFMConfigurationError,
    JointFMHTTPStatusError,
    JointFMInstanceSettings,
    JointFMRetryConfig,
    JointFMSettings,
    cli,
    run_max_utilization_check,
)
from jointfm_client.configuration import DEFAULT_RETRY_STATUS_CODES

_MODEL_VERSION = "jointfm-inference:0.3.0+ckpt.sdk-test"
_DATA_GENERATION = {
    "sampler_type": "studentt",
    "min_series": 1,
    "max_series": 3,
    "t_input": 10.0,
    "t_output": 3.0,
    "n_input": 5,
    "n_output": 2,
}
_SAMPLE_CAP = 2
_BARRIER_TIMEOUT_SECONDS = 5.0


def _url(deployment_id: str) -> str:
    """Hosted predict URL for one deployment ID."""
    return (
        "https://app.datarobot.com/api/v2/deployments/"
        f"{deployment_id}/predictionsUnstructured"
    )


def _settings(*deployment_ids: str) -> JointFMSettings:
    """Hosted settings for one endpoint or a deployment-ID pool."""
    primary = _url(deployment_ids[0])
    return JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url=primary,
        predict_url=primary,
        deployment_selector=(
            "deployment_id" if len(deployment_ids) == 1 else "deployment_ids"
        ),
        schema_version="v5",
        instances=tuple(
            JointFMInstanceSettings(
                deployment_id=deployment_id, predict_url=_url(deployment_id)
            )
            for deployment_id in deployment_ids
        )
        if len(deployment_ids) > 1
        else (),
        model_version=_MODEL_VERSION,
        deployment_id=deployment_ids[0],
    )


def _health_payload(
    *,
    data_generation: bool = True,
    max_sample_count: int = _SAMPLE_CAP,
    max_concurrent_requests: int = 1,
) -> dict[str, Any]:
    """Health payload advertising a small capacity envelope."""
    payload: dict[str, Any] = {
        "status": "ok",
        "schema_version": "v5",
        "image_version": "0.3.0",
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "sdk-test",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "studentt",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": ["forecast"],
        "supported_condition_kinds": [],
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "max_sample_count": max_sample_count,
        "max_concurrent_requests": max_concurrent_requests,
    }
    if data_generation:
        payload["data_generation"] = dict(_DATA_GENERATION)
    return payload


class _EnvelopeTransport:
    """Serve health and echo-shaped sample forecasts; fail chosen deployments.

    ``sample_caps`` and ``concurrency`` set one deployment's advertised
    ``max_sample_count`` and ``max_concurrent_requests``. With
    ``simultaneous_after`` set, every forecast after the first that many must
    reach a barrier together with ``barrier_parties - 1`` others, so a stage
    that sends them one at a time breaks the barrier instead of passing.
    """

    def __init__(
        self,
        *,
        data_generation: bool = True,
        forecast_failures: Mapping[str, int] | None = None,
        health_failures: frozenset[str] = frozenset(),
        sample_caps: Mapping[str, int] | None = None,
        concurrency: Mapping[str, int] | None = None,
        simultaneous_after: int | None = None,
        barrier_parties: int = 1,
    ) -> None:
        self.data_generation = data_generation
        self.forecast_failures = dict(forecast_failures or {})
        self.health_failures = health_failures
        self.sample_caps = dict(sample_caps or {})
        self.concurrency = dict(concurrency or {})
        self.simultaneous_after = simultaneous_after
        self.barrier = threading.Barrier(
            barrier_parties, timeout=_BARRIER_TIMEOUT_SECONDS
        )
        self.lock = threading.Lock()
        self.forecasts: list[tuple[str, Mapping[str, Any]]] = []

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        raise AssertionError(f"unexpected GET {url}")

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        deployment_id = url.rstrip("/").split("/")[-2]
        if payload.get("request_type") == "health":
            if deployment_id in self.health_failures:
                raise _http_error(deployment_id, next(iter(DEFAULT_RETRY_STATUS_CODES)))
            return _health_payload(
                data_generation=self.data_generation,
                max_sample_count=self.sample_caps.get(deployment_id, _SAMPLE_CAP),
                max_concurrent_requests=self.concurrency.get(deployment_id, 1),
            )
        with self.lock:
            self.forecasts.append((deployment_id, payload))
            forecast_index = len(self.forecasts) - 1
        if (
            self.simultaneous_after is not None
            and forecast_index >= self.simultaneous_after
        ):
            self.barrier.wait()
        if deployment_id in self.forecast_failures:
            raise _http_error(deployment_id, self.forecast_failures[deployment_id])
        query_times = cast(list[int], payload["query_times"])
        requested_columns = cast(list[str], payload["requested_columns"])
        samples = [
            [[float(sample)] * len(requested_columns) for _ in query_times]
            for sample in range(cast(int, payload["n_samples"]))
        ]
        return {
            "schema_version": "v5",
            "image_version": "0.3.0",
            "model_version": _MODEL_VERSION,
            "checkpoint_version": "sdk-test",
            "head": "studentt",
            "query_mode": "forecast",
            "return_mode": "samples",
            "outputs": {
                "query_times": query_times,
                "requested_columns": requested_columns,
                "mean": None,
                "samples": samples,
                "quantiles": None,
            },
            "diagnostics": {
                "history_rows": len(cast(list[Any], payload["history_rows"])),
                "horizon_count": len(query_times),
                "seed": payload.get("seed"),
            },
            "errors": [],
        }


def _http_error(deployment_id: str, status_code: int) -> JointFMHTTPStatusError:
    """HTTP status error raised by a failing fake deployment."""
    return JointFMHTTPStatusError(
        f"{deployment_id} failed",
        status_code=status_code,
        response_body_excerpt="failed",
    )


def _client(transport: _EnvelopeTransport, *deployment_ids: str) -> JointFMClient:
    """Non-retrying client over the fake transport."""
    return JointFMClient(
        settings=_settings(*deployment_ids),
        transport=transport,
        retry_config=JointFMRetryConfig(max_attempts=1),
    )


def test_single_endpoint_sends_the_full_envelope() -> None:
    """One endpoint receives max_series targets, n_input rows, n_output steps, and the cap."""
    transport = _EnvelopeTransport()

    report = run_max_utilization_check(_client(transport, "single-id"))

    assert report.passed
    assert report.request_slots == 1
    assert report.concurrent_saturation is None
    assert report.single_request.n_samples == _SAMPLE_CAP
    assert report.single_request.request_count == 1
    [(deployment_id, payload)] = transport.forecasts
    assert deployment_id == "single-id"
    assert payload["n_samples"] == _SAMPLE_CAP
    assert len(payload["history_rows"]) == _DATA_GENERATION["n_input"]
    assert len(payload["query_times"]) == _DATA_GENERATION["n_output"]
    assert len(payload["requested_columns"]) == _DATA_GENERATION["max_series"]
    assert {column["role"] for column in payload["columns"]} == {"target"}
    assert report.format_lines()[-1] == "VERDICT: PASS"


def test_saturation_fills_every_request_slot_of_one_endpoint_at_once() -> None:
    """An endpoint serving two requests at once receives two maximal requests together."""
    transport = _EnvelopeTransport(
        concurrency={"single-id": 2}, simultaneous_after=1, barrier_parties=2
    )

    report = run_max_utilization_check(_client(transport, "single-id"))

    assert report.passed
    assert report.request_slots == 2
    saturation = report.concurrent_saturation
    assert saturation is not None
    assert saturation.request_count == 2
    assert saturation.n_samples == 2 * _SAMPLE_CAP
    burst = transport.forecasts[1:]
    assert [deployment_id for deployment_id, _ in burst] == ["single-id"] * 2
    assert sorted(payload["seed"] for _, payload in burst) == [7, 8]
    assert all(payload["n_samples"] == _SAMPLE_CAP for _, payload in burst)


def test_pool_saturation_sends_each_endpoint_its_slots_and_own_cap() -> None:
    """Every pool slot is filled at once, each with its endpoint's own sample cap."""
    transport = _EnvelopeTransport(
        sample_caps={"a-id": 2, "b-id": 3},
        concurrency={"a-id": 2},
        simultaneous_after=1,
        barrier_parties=3,
    )

    report = run_max_utilization_check(_client(transport, "a-id", "b-id"))

    assert report.passed
    assert report.single_request.n_samples == 2
    saturation = report.concurrent_saturation
    assert saturation is not None
    assert saturation.request_count == 3
    assert saturation.n_samples == 2 + 2 + 3
    burst = sorted(
        (deployment_id, payload["n_samples"])
        for deployment_id, payload in transport.forecasts[1:]
    )
    assert burst == [("a-id", 2), ("a-id", 2), ("b-id", 3)]


def test_every_failed_slot_is_reported() -> None:
    """A burst records one error per failed request, not only the first."""
    transport = _EnvelopeTransport(
        forecast_failures={"single-id": 500}, concurrency={"single-id": 2}
    )

    report = run_max_utilization_check(_client(transport, "single-id"))

    saturation = report.concurrent_saturation
    assert saturation is not None
    assert len(saturation.errors) == 2
    assert all("JointFMHTTPStatusError" in error for error in saturation.errors)
    assert not report.passed


def test_pool_failover_fails_the_stage_although_the_forecast_succeeds() -> None:
    """A request rerouted to a healthy peer still fails the verdict."""
    retryable = next(iter(DEFAULT_RETRY_STATUS_CODES))
    transport = _EnvelopeTransport(forecast_failures={"a-id": retryable})

    report = run_max_utilization_check(_client(transport, "a-id", "b-id"))

    assert report.single_request.errors == ()
    assert report.single_request.failovers
    assert not report.passed
    assert report.format_lines()[-1] == "VERDICT: FAIL"


def test_service_error_is_recorded_as_a_failed_stage() -> None:
    """A non-retryable service error ends the stage and fails the verdict."""
    transport = _EnvelopeTransport(forecast_failures={"single-id": 500})

    report = run_max_utilization_check(_client(transport, "single-id"))

    [error] = report.single_request.errors
    assert "JointFMHTTPStatusError" in error
    assert not report.passed


def test_unavailable_pool_endpoint_fails_the_verdict() -> None:
    """An endpoint that fails the health probe fails the check."""
    transport = _EnvelopeTransport(health_failures=frozenset({"b-id"}))

    report = run_max_utilization_check(_client(transport, "a-id", "b-id"))

    assert report.unavailable_instances == ("b-id",)
    assert not report.passed


@pytest.mark.parametrize(
    ("deployment_ids", "unknown_id"),
    [(("single-id",), "other-id"), (("a-id", "b-id"), "c-id")],
)
def test_endpoint_client_rejects_an_unconfigured_deployment(
    deployment_ids: tuple[str, ...], unknown_id: str
) -> None:
    """A slot client is only built for a deployment the client already targets."""
    client = _client(_EnvelopeTransport(), *deployment_ids)

    with pytest.raises(JointFMConfigurationError, match=unknown_id):
        client.endpoint_client(unknown_id)


def test_pool_endpoint_client_targets_only_its_instance() -> None:
    """A pool instance's client posts to that instance alone, never through the pool."""
    transport = _EnvelopeTransport()
    endpoint = _client(transport, "a-id", "b-id").endpoint_client("b-id")

    assert endpoint.settings is not None
    assert endpoint.settings.instances == ()
    assert endpoint.predict_url == _url("b-id")
    assert endpoint.health(cache=True).max_concurrent_requests == 1


def test_retrying_client_is_rejected() -> None:
    """Retries would hide failures at maximum load, so the check refuses them."""
    client = JointFMClient(
        settings=_settings("single-id"),
        transport=_EnvelopeTransport(),
        retry_config=JointFMRetryConfig(max_attempts=2),
    )

    with pytest.raises(JointFMConfigurationError, match="max_attempts=1"):
        run_max_utilization_check(client)


def test_missing_data_generation_block_is_rejected() -> None:
    """Without an advertised envelope there is nothing to size the request from."""
    transport = _EnvelopeTransport(data_generation=False)

    with pytest.raises(JointFMCapacityError, match="data_generation"):
        run_max_utilization_check(_client(transport, "single-id"))


def test_cli_exit_code_follows_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI builds a non-retrying client and exits 1 on a failed verdict."""
    built_with: list[JointFMRetryConfig] = []

    def _from_env(*, dotenv_path: object, retry_config: JointFMRetryConfig) -> Any:
        del dotenv_path
        built_with.append(retry_config)
        return _client(
            _EnvelopeTransport(forecast_failures={"single-id": 500}), "single-id"
        )

    monkeypatch.setattr(cli.JointFMClient, "from_env", _from_env)
    stdout = io.StringIO()
    args = cli._build_parser().parse_args(["max-utilization", "--no-dotenv"])

    assert args.handler(args, stdout) == 1
    assert built_with[0].max_attempts == 1
    assert stdout.getvalue().rstrip().endswith("VERDICT: FAIL")
