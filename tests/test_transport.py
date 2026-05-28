from __future__ import annotations

from collections.abc import Mapping
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter

import jointfm_client.client as client_module
from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    ForecastResponse,
    HealthMetadata,
    JointFMClient,
    JointFMConfigurationError,
    JointFMRequestEncodingError,
    JointFMRequestError,
    JointFMHTTPStatusError,
    JointFMHTTPTransport,
    JointFMResponseDecodeError,
    JointFMServiceError,
    JointFMRetryConfig,
    JointFMSettings,
    JointFMTimeoutConfig,
    MeanForecastResult,
    SampleForecastResult,
    UnsupportedModelVersionError,
)


class StaticResponseAdapter(BaseAdapter):
    def __init__(self, response: requests.Response) -> None:
        super().__init__()
        self.response = response
        self.requests: list[requests.PreparedRequest] = []
        self.kwargs: list[dict[str, Any]] = []

    def send(
        self,
        request,
        stream=False,
        timeout=None,
        verify=True,
        cert=None,
        proxies=None,
    ) -> requests.Response:
        self.requests.append(request)
        self.kwargs.append(
            {
                "stream": stream,
                "timeout": timeout,
                "verify": verify,
                "cert": cert,
                "proxies": proxies,
            }
        )
        self.response.request = request
        self.response.url = request.url or ""
        return self.response

    def close(self) -> None:
        return None


class RecordingTransport:
    def __init__(self) -> None:
        self.health_url: str | None = None
        self.predict_url: str | None = None
        self.payload: Mapping[str, Any] | None = None
        self.get_count = 0
        self.post_count = 0
        self.health_payload: Mapping[str, Any] = _health_payload()
        self.predict_payload: Mapping[str, Any] = _forecast_response_payload()

    def get_json(self, url: str) -> Mapping[str, Any]:
        self.health_url = url
        self.get_count += 1
        return self.health_payload

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.predict_url = url
        self.payload = payload
        self.post_count += 1
        return self.predict_payload


class ClosingSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


class ErroringSession(requests.Session):
    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        del args, kwargs
        raise requests.ConnectionError("boom")


def _health_payload(*, model_version: str = "jointfm-inference:0.2.0+ckpt.sdk-test") -> dict[str, object]:
    return {
        "status": "ok",
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": model_version,
        "checkpoint_version": "sdk-test",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "studentt",
        "supported_query_modes": ["forecast"],
        "supported_return_modes": ["mean", "samples", "quantiles"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
    }


def _forecast_response_payload(*, return_mode: str = "mean") -> dict[str, object]:
    outputs: dict[str, object] = {
        "query_times": [2],
        "requested_columns": ["target"],
        "mean": [[12.0]] if return_mode == "mean" else None,
        "samples": [[[12.0]]] if return_mode == "samples" else None,
        "quantiles": [
            {"quantile": 0.1, "values": [[11.0]]},
            {"quantile": 0.9, "values": [[13.0]]},
        ]
        if return_mode == "quantiles"
        else None,
    }
    return {
        "schema_version": "v1",
        "image_version": "0.2.0",
        "model_version": "jointfm-inference:0.2.0+ckpt.sdk-test",
        "checkpoint_version": "sdk-test",
        "head": "studentt",
        "query_mode": "forecast",
        "return_mode": return_mode,
        "outputs": outputs,
        "diagnostics": {"history_rows": 2, "horizon_count": 1, "seed": 7},
        "errors": [],
    }


def _response(
    *,
    status_code: int = HTTPStatus.OK,
    body: bytes,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    response.headers.update(headers or {})
    return response


def test_transport_posts_json_with_headers_timeout_and_user_agent() -> None:
    session = requests.Session()
    adapter = StaticResponseAdapter(_response(body=b'{"ok": true}'))
    transport = JointFMHTTPTransport(
        session=session,
        headers={
            "Authorization": "Bearer secret-token",
            "Accept": "application/json",
            "Content-Type": "application/json;charset=UTF-8",
        },
        timeout=JointFMTimeoutConfig(connect_seconds=1.5, read_seconds=2.5),
        retry_config=JointFMRetryConfig(max_attempts=1),
    )
    session.mount("https://", adapter)

    result = transport.post_json("https://example.com/predict", {"schema_version": "v1"})

    assert result == {"ok": True}
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    request_headers = dict(request.headers or {})
    assert request_headers["Authorization"] == "Bearer secret-token"
    assert request_headers["Accept"] == "application/json"
    assert request_headers["Content-Type"] == "application/json;charset=UTF-8"
    assert request_headers["User-Agent"] == "jointfm-client/0.0.1"
    assert adapter.kwargs[0]["timeout"] == (1.5, 2.5)
    request_body = request.body
    assert isinstance(request_body, bytes)
    assert json.loads(request_body.decode("utf-8")) == {"schema_version": "v1"}


def test_transport_from_settings_attaches_hosted_auth_headers_and_closes_session() -> None:
    session = ClosingSession()
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url="https://app.datarobot.com/api/v2/deployments/deployment-id/healthz",
        predict_url=(
            "https://app.datarobot.com/api/v2/deployments/"
            "deployment-id/predictionsUnstructured"
        ),
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )
    transport = JointFMHTTPTransport.from_settings(session=session, settings=settings)
    adapter = StaticResponseAdapter(_response(body=b'{"ok": true}'))
    session.mount("https://", adapter)

    result = transport.get_json("https://example.com/healthz")

    assert result == {"ok": True}
    assert dict(adapter.requests[0].headers or {})["Authorization"] == "Bearer secret-token"
    transport.close()
    assert session.closed is True


def test_transport_from_local_settings_omits_hosted_auth_headers() -> None:
    session = requests.Session()
    settings = JointFMSettings(
        datarobot_endpoint=None,
        datarobot_api_token=None,
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        deployment_selector="local_service",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.local-test",
        local_base_url="http://127.0.0.1:8080",
    )
    transport = JointFMHTTPTransport.from_settings(session=session, settings=settings)
    adapter = StaticResponseAdapter(_response(body=b'{"ok": true}'))
    session.mount("http://", adapter)

    result = transport.get_json("http://127.0.0.1:8080/healthz")

    assert result == {"ok": True}
    request_headers = dict(adapter.requests[0].headers or {})
    assert "Authorization" not in request_headers
    assert request_headers["User-Agent"] == "jointfm-client/0.0.1"


def test_transport_retries_retryable_server_responses() -> None:
    server, handler = _start_json_server([HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.OK])
    try:
        transport = JointFMHTTPTransport(retry_config=JointFMRetryConfig(max_attempts=2))

        result = transport.post_json(_server_url(server), {"schema_version": "v1"})

        assert result == {"ok": True}
        assert handler.request_count == 2
    finally:
        server.shutdown()
        server.server_close()


def test_retry_config_builds_expected_urllib3_policy() -> None:
    retry = JointFMRetryConfig(
        max_attempts=4,
        backoff_seconds=0.5,
        status_codes=(408, 500),
    ).as_urllib3_retry()

    assert retry.total == 3
    assert retry.connect == 3
    assert retry.read == 3
    assert retry.status == 3
    assert retry.backoff_factor == 0.5
    assert retry.status_forcelist == (408, 500)


def test_transport_does_not_retry_validation_errors() -> None:
    server, handler = _start_json_server([HTTPStatus.BAD_REQUEST, HTTPStatus.OK])
    try:
        transport = JointFMHTTPTransport(retry_config=JointFMRetryConfig(max_attempts=2))

        with pytest.raises(JointFMHTTPStatusError) as exc_info:
            transport.post_json(_server_url(server), {"schema_version": "v1"})

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert exc_info.value.datarobot_request_id == "request-id-1"
        assert exc_info.value.jointfm_errors == (
            {
                "code": "VALIDATION_ERROR",
                "message": "bad request",
                "field": "schema_version",
            },
        )
        assert "bad request" in str(exc_info.value)
        assert "VALIDATION_ERROR" in exc_info.value.response_body_excerpt
        assert handler.request_count == 1
    finally:
        server.shutdown()
        server.server_close()


def test_transport_rejects_non_json_serializable_payloads() -> None:
    transport = JointFMHTTPTransport(retry_config=JointFMRetryConfig(max_attempts=1))

    with pytest.raises(JointFMRequestEncodingError, match="JSON-serializable"):
        transport.post_json(
            "https://example.com/predict",
            {"schema_version": "v1", "bad": object()},
        )


def test_transport_wraps_request_exceptions() -> None:
    transport = JointFMHTTPTransport(
        session=ErroringSession(),
        retry_config=JointFMRetryConfig(max_attempts=1),
    )

    with pytest.raises(JointFMRequestError, match="boom"):
        transport.get_json("https://example.com/healthz")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"", "empty response"),
        (b"not-json", "non-JSON"),
        (b"[]", "not an object"),
    ],
)
def test_transport_rejects_malformed_json_responses(body: bytes, message: str) -> None:
    session = requests.Session()
    adapter = StaticResponseAdapter(
        _response(
            body=body,
            headers={"X-DataRobot-Request-ID": "decode-request-id"},
        )
    )
    transport = JointFMHTTPTransport(
        session=session,
        retry_config=JointFMRetryConfig(max_attempts=1),
    )
    session.mount("https://", adapter)

    with pytest.raises(JointFMResponseDecodeError, match=message) as exc_info:
        transport.get_json("https://example.com/healthz")

    assert exc_info.value.status_code == HTTPStatus.OK
    assert exc_info.value.datarobot_request_id == "decode-request-id"


def test_timeout_and_retry_config_reject_invalid_values() -> None:
    with pytest.raises(JointFMConfigurationError, match="connect_seconds"):
        JointFMTimeoutConfig(connect_seconds=0)

    with pytest.raises(JointFMConfigurationError, match="max_attempts"):
        JointFMRetryConfig(max_attempts=0)

    with pytest.raises(JointFMConfigurationError, match="status_codes"):
        JointFMRetryConfig(status_codes=(200,))


def test_client_predict_uses_configured_transport_and_settings() -> None:
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url="https://app.datarobot.com/api/v2/deployments/deployment-id/healthz",
        predict_url="https://app.datarobot.com/api/v2/deployments/deployment-id/predictionsUnstructured",
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )
    transport = RecordingTransport()
    client = JointFMClient(settings=settings, transport=transport)
    payload = {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.sdk-test",
    }

    result = client.predict(payload)

    assert result == _forecast_response_payload()
    assert transport.predict_url == settings.predict_url
    assert transport.payload is payload


def test_client_health_returns_typed_metadata_and_caches_only_when_requested() -> None:
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url="https://app.datarobot.com/api/v2/deployments/deployment-id/healthz",
        predict_url="https://app.datarobot.com/api/v2/deployments/deployment-id/predictionsUnstructured",
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )
    transport = RecordingTransport()
    client = JointFMClient(settings=settings, transport=transport)

    uncached = client.health()
    uncached_again = client.health()
    cached = client.health(cache=True)
    cached_again = client.health(cache=True)
    refreshed = client.health(cache=True, refresh=True)

    assert isinstance(uncached, HealthMetadata)
    assert uncached.model_version == settings.model_version
    assert uncached_again.model_version == settings.model_version
    assert cached is cached_again
    assert refreshed.model_version == settings.model_version
    assert transport.health_url == settings.health_url
    assert transport.get_count == 4


def test_client_from_env_forwards_timeout_and_retry_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout: JointFMTimeoutConfig | None = None
    captured_retry_config: JointFMRetryConfig | None = None
    transport = RecordingTransport()

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
    ) -> RecordingTransport:
        del settings, session, user_agent, response_body_excerpt_characters
        del datarobot_request_id_headers
        nonlocal captured_timeout, captured_retry_config
        captured_timeout = timeout
        captured_retry_config = retry_config
        return transport

    monkeypatch.setattr(
        client_module.JointFMHTTPTransport,
        "from_settings",
        capture_transport,
    )
    timeout = JointFMTimeoutConfig(connect_seconds=1.0, read_seconds=2.0)
    retry_config = JointFMRetryConfig(max_attempts=1)

    client = JointFMClient.from_env(
        env={
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "secret-token",
            "JOINTFM_DEPLOYMENT_ID": "deployment-id",
            "JOINTFM_SCHEMA_VERSION": "v1",
            "JOINTFM_MODEL_VERSION": "jointfm-inference:0.2.0+ckpt.sdk-test",
        },
        dotenv_path=None,
        timeout=timeout,
        retry_config=retry_config,
    )

    client.health()

    assert captured_timeout is timeout
    assert captured_retry_config is retry_config


def test_client_health_rejects_cached_model_mismatch() -> None:
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url="https://app.datarobot.com/api/v2/deployments/deployment-id/healthz",
        predict_url="https://app.datarobot.com/api/v2/deployments/deployment-id/predictionsUnstructured",
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )
    transport = RecordingTransport()
    transport.health_payload = _health_payload(
        model_version="jointfm-inference:9.9.9+ckpt.other"
    )
    client = JointFMClient(settings=settings, transport=transport)

    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        client.health(cache=True)


def test_client_forecast_builds_payload_from_rows_and_returns_typed_response() -> None:
    transport = RecordingTransport()
    client = JointFMClient(
        predict_url="http://localhost:8080/predict",
        transport=transport,
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )

    result = client.forecast(
        [{"target": 10.0}, {"target": 11.0}],
        schema=schema,
        query_times=[2],
        requested_columns=["target"],
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        seed=7,
    )

    assert isinstance(result, ForecastResponse)
    assert isinstance(result, MeanForecastResult)
    assert result.mean == ((12.0,),)
    assert result.outputs.mean == ((12.0,),)
    assert transport.predict_url == "http://localhost:8080/predict"
    assert transport.payload == {
        "schema_version": "v1",
        "model_version": "jointfm-inference:0.2.0+ckpt.sdk-test",
        "query_mode": "forecast",
        "return_mode": "mean",
        "time_index_mode": "ordinal",
        "columns": [{"name": "target", "modality": "numeric", "role": "target"}],
        "history_rows": [{"target": 10.0}, {"target": 11.0}],
        "query_times": [2],
        "requested_columns": ["target"],
        "seed": 7,
    }


def test_client_forecast_samples_batches_when_service_reports_sample_cap() -> None:
    class SampleCapTransport:
        def __init__(self) -> None:
            self.payloads: list[Mapping[str, Any]] = []
            self.sample_offset = 0

        def get_json(self, url: str) -> Mapping[str, Any]:
            raise AssertionError(f"unexpected health request: {url}")

        def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
            assert url == "http://localhost:8080/predict"
            self.payloads.append(dict(payload))
            if len(self.payloads) == 1:
                raise JointFMHTTPStatusError(
                    "JointFM service returned HTTP 470",
                    status_code=470,
                    response_body_excerpt=(
                        '{"message":"{\\"message\\": '
                        '\\"Container response not 200 for k8s service route.\\", '
                        '\\"container_response\\": {\\"message\\": '
                        '\\"ERROR: n_samples exceeds the configured container cap: '
                        'requested 5, max 2\\"}}"}'
                    ),
                    datarobot_request_id="request-id",
                )

            sample_count = payload["n_samples"]
            assert isinstance(sample_count, int)
            samples = [
                [[float(sample_index)]]
                for sample_index in range(
                    self.sample_offset,
                    self.sample_offset + sample_count,
                )
            ]
            self.sample_offset += sample_count
            response_payload = _forecast_response_payload(return_mode="samples")
            outputs = response_payload["outputs"]
            assert isinstance(outputs, dict)
            outputs["samples"] = samples
            diagnostics = response_payload["diagnostics"]
            assert isinstance(diagnostics, dict)
            diagnostics["seed"] = payload.get("seed")
            return response_payload

    transport = SampleCapTransport()
    client = JointFMClient(
        predict_url="http://localhost:8080/predict",
        transport=transport,
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )

    result = client.forecast_samples(
        [{"target": 10.0}, {"target": 11.0}],
        schema=schema,
        query_times=[2],
        requested_columns=["target"],
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        n_samples=5,
        seed=7,
    )

    assert isinstance(result, SampleForecastResult)
    assert result.samples == (
        ((0.0,),),
        ((1.0,),),
        ((2.0,),),
        ((3.0,),),
        ((4.0,),),
    )
    assert result.diagnostics.seed == 7
    assert [payload["n_samples"] for payload in transport.payloads] == [5, 2, 2, 1]
    assert [payload["seed"] for payload in transport.payloads] == [7, 7, 8, 9]

    second_result = client.forecast_samples(
        [{"target": 10.0}, {"target": 11.0}],
        schema=schema,
        query_times=[2],
        requested_columns=["target"],
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        n_samples=3,
        seed=11,
    )

    assert second_result.samples == (
        ((5.0,),),
        ((6.0,),),
        ((7.0,),),
    )
    assert [payload["n_samples"] for payload in transport.payloads] == [
        5,
        2,
        2,
        1,
        2,
        1,
    ]
    assert [payload["seed"] for payload in transport.payloads] == [
        7,
        7,
        8,
        9,
        11,
        12,
    ]


def test_client_predict_raises_typed_service_error_for_success_payload_errors() -> None:
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url="https://app.datarobot.com/api/v2/deployments/deployment-id/healthz",
        predict_url="https://app.datarobot.com/api/v2/deployments/deployment-id/predictionsUnstructured",
        deployment_selector="deployment_id",
        schema_version="v1",
        model_version="jointfm-inference:0.2.0+ckpt.sdk-test",
        deployment_id="deployment-id",
    )
    transport = RecordingTransport()
    transport.predict_payload = {
        "schema_version": "v1",
        "errors": [
            {
                "code": "VALIDATION_ERROR",
                "message": "bad request",
                "field": "query_times",
            }
        ],
    }
    client = JointFMClient(settings=settings, transport=transport)

    with pytest.raises(JointFMServiceError) as exc_info:
        client.predict(
            {
                "schema_version": "v1",
                "model_version": settings.model_version,
            }
        )

    assert exc_info.value.jointfm_errors == (
        {
            "code": "VALIDATION_ERROR",
            "message": "bad request",
            "field": "query_times",
        },
    )


def test_client_forecast_uses_cached_health_model_version_and_validates_mismatch() -> None:
    transport = RecordingTransport()
    client = JointFMClient(
        health_url="http://localhost:8080/healthz",
        predict_url="http://localhost:8080/predict",
        transport=transport,
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )

    client.health(cache=True)
    result = client.forecast(
        [{"target": 10.0}, {"target": 11.0}],
        schema=schema,
        query_times=[2],
        requested_columns=["target"],
    )

    assert result.model_version == "jointfm-inference:0.2.0+ckpt.sdk-test"
    assert transport.payload is not None
    assert transport.payload["model_version"] == "jointfm-inference:0.2.0+ckpt.sdk-test"
    with pytest.raises(UnsupportedModelVersionError, match="model_version"):
        client.forecast(
            [{"target": 10.0}, {"target": 11.0}],
            schema=schema,
            query_times=[2],
            model_version="jointfm-inference:9.9.9+ckpt.other",
        )


def test_client_forecast_requires_model_version_without_settings_or_health() -> None:
    client = JointFMClient(
        predict_url="http://localhost:8080/predict",
        transport=RecordingTransport(),
    )
    schema = DataFrameSchema(
        columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
        time_index_mode="ordinal",
    )

    with pytest.raises(JointFMConfigurationError, match="model_version"):
        client.forecast(
            [{"target": 10.0}, {"target": 11.0}],
            schema=schema,
            query_times=[2],
        )


def _start_json_server(
    statuses: list[HTTPStatus],
) -> tuple[ThreadingHTTPServer, type[BaseHTTPRequestHandler]]:
    class JSONHandler(BaseHTTPRequestHandler):
        request_count = 0

        def do_POST(self) -> None:
            type(self).request_count += 1
            status = statuses[type(self).request_count - 1]
            if status == HTTPStatus.OK:
                payload = {"ok": True}
            else:
                payload = {
                    "schema_version": "v1",
                    "errors": [
                        {
                            "code": "VALIDATION_ERROR",
                            "message": "bad request",
                            "field": "schema_version",
                        }
                    ],
                }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-DataRobot-Request-ID", f"request-id-{type(self).request_count}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), JSONHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, JSONHandler


def _server_url(server: ThreadingHTTPServer) -> str:
    host = server.server_address[0]
    port = server.server_address[1]
    return f"http://{host}:{port}/predict"