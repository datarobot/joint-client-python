from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    JointFMClient,
    JointFMServiceError,
    JointFMSettings,
)


class _SurfaceHandlerBase(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]]
    route_map: Mapping[
        tuple[str, str],
        tuple[HTTPStatus, Mapping[str, Any]]
        | Mapping[str | None, tuple[HTTPStatus, Mapping[str, Any]]],
    ]


def test_hosted_surface_uses_datarobot_routes_and_auth_headers(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    request_payload = json_fixture_loader("forecast_mean_request")
    health_payload = json_fixture_loader("health_metadata")
    response_payload = json_fixture_loader("forecast_mean_response")
    predict_path = "/deployments/deployment-id/predictionsUnstructured"
    server, handler = _start_surface_server(
        {
            ("POST", predict_path): {
                "health": (HTTPStatus.OK, health_payload),
                None: (HTTPStatus.OK, response_payload),
            },
        }
    )
    try:
        base_url = _server_base_url(server)
        predict_url = f"{base_url}{predict_path}"
        settings = JointFMSettings(
            datarobot_endpoint="https://app.datarobot.com/api/v2",
            datarobot_api_token="secret-token",
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="deployment_id",
            schema_version="v1",
            model_version=request_payload["model_version"],
            deployment_id="deployment-id",
        )
        client = JointFMClient(settings=settings)

        health = client.health(cache=True)
        response = client.predict(request_payload)

        assert health.model_version == settings.model_version
        assert response == response_payload
        assert [entry["path"] for entry in handler.requests] == [predict_path, predict_path]
        assert [entry["method"] for entry in handler.requests] == ["POST", "POST"]
        assert handler.requests[0]["body"] == {"request_type": "health"}
        assert handler.requests[0]["headers"]["Authorization"] == "Bearer secret-token"
        assert handler.requests[1]["headers"]["Accept"] == "*/*"
        assert (
            handler.requests[1]["headers"]["Content-Type"]
            == "application/json;charset=UTF-8"
        )
        assert handler.requests[1]["body"] == request_payload
    finally:
        server.shutdown()
        server.server_close()


def test_local_surface_uses_direct_service_routes_without_hosted_auth(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    request_payload = json_fixture_loader("forecast_mean_request")
    health_payload = json_fixture_loader("health_metadata")
    response_payload = json_fixture_loader("forecast_mean_response")
    server, handler = _start_surface_server(
        {
            ("GET", "/healthz"): (HTTPStatus.OK, health_payload),
            ("POST", "/predict"): (HTTPStatus.OK, response_payload),
        }
    )
    try:
        base_url = _server_base_url(server)
        client = JointFMClient(
            health_url=f"{base_url}/healthz",
            predict_url=f"{base_url}/predict",
        )
        schema = DataFrameSchema(
            columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
            time_index_mode="ordinal",
        )

        health = client.health(cache=True)
        result = client.forecast_mean(
            [{"target": 10.0}, {"target": 11.0}],
            schema=schema,
            query_times=[2],
            requested_columns=["target"],
            seed=7,
        )

        assert health.model_version == request_payload["model_version"]
        assert result.mean == ((12.0,),)
        assert [entry["path"] for entry in handler.requests] == ["/healthz", "/predict"]
        assert "Authorization" not in handler.requests[0]["headers"]
        assert "Authorization" not in handler.requests[1]["headers"]
        assert handler.requests[1]["body"] == request_payload
    finally:
        server.shutdown()
        server.server_close()


def test_local_surface_rejects_samples_below_requested_lower_bound(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    response_payload = json_fixture_loader("forecast_samples_response")
    response_payload["outputs"]["samples"] = [[[-0.5]], [[12.5]]]
    server, handler = _start_surface_server(
        {("POST", "/predict"): (HTTPStatus.OK, response_payload)}
    )
    try:
        base_url = _server_base_url(server)
        client = JointFMClient(predict_url=f"{base_url}/predict")
        schema = DataFrameSchema(
            columns=(
                ColumnSpec(
                    name="target",
                    modality="numeric",
                    role="target",
                    lower_bound=0.0,
                ),
            ),
            time_index_mode="ordinal",
        )

        with pytest.raises(JointFMServiceError, match="lower_bound"):
            client.forecast_samples(
                [{"target": 10.0}, {"target": 11.0}],
                schema=schema,
                query_times=[2],
                requested_columns=["target"],
                model_version=response_payload["model_version"],
                n_samples=2,
                seed=7,
            )

        assert handler.requests[0]["body"]["columns"][0]["lower_bound"] == 0.0
    finally:
        server.shutdown()
        server.server_close()


def test_hosted_surface_auto_discovers_model_version_when_settings_unpinned(
    json_fixture_loader: Callable[[str], dict[str, Any]],
) -> None:
    request_payload = json_fixture_loader("forecast_mean_request")
    health_payload = json_fixture_loader("health_metadata")
    response_payload = json_fixture_loader("forecast_mean_response")
    predict_path = "/deployments/deployment-id/predictionsUnstructured"
    server, handler = _start_surface_server(
        {
            ("POST", predict_path): {
                "health": (HTTPStatus.OK, health_payload),
                None: (HTTPStatus.OK, response_payload),
            },
        }
    )
    try:
        base_url = _server_base_url(server)
        predict_url = f"{base_url}{predict_path}"
        settings = JointFMSettings(
            datarobot_endpoint="https://app.datarobot.com/api/v2",
            datarobot_api_token="secret-token",
            health_url=predict_url,
            predict_url=predict_url,
            deployment_selector="deployment_id",
            schema_version="v1",
            deployment_id="deployment-id",
        )
        assert settings.model_version is None
        client = JointFMClient(settings=settings)
        schema = DataFrameSchema(
            columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
            time_index_mode="ordinal",
        )

        result = client.forecast_mean(
            [{"target": 10.0}, {"target": 11.0}],
            schema=schema,
            query_times=[2],
            requested_columns=["target"],
            seed=7,
        )

        assert result.model_version == health_payload["model_version"]
        assert [entry["body"].get("request_type") for entry in handler.requests] == [
            "health",
            None,
        ]
        assert handler.requests[1]["body"]["model_version"] == health_payload["model_version"]
    finally:
        server.shutdown()
        server.server_close()


def _start_surface_server(
    routes: Mapping[
        tuple[str, str],
        tuple[HTTPStatus, Mapping[str, Any]]
        | Mapping[str | None, tuple[HTTPStatus, Mapping[str, Any]]],
    ],
) -> tuple[ThreadingHTTPServer, type[_SurfaceHandlerBase]]:
    class SurfaceHandler(_SurfaceHandlerBase):
        requests: list[dict[str, Any]] = []
        route_map = routes

        def do_GET(self) -> None:
            self._handle_request("GET")

        def do_POST(self) -> None:
            self._handle_request("POST")

        def _handle_request(self, method: str) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_length) if content_length > 0 else b""
            decoded_body = None if request_body == b"" else json.loads(request_body.decode("utf-8"))
            type(self).requests.append(
                {
                    "method": method,
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": decoded_body,
                }
            )
            route_value = type(self).route_map[(method, self.path)]
            if isinstance(route_value, Mapping):
                discriminator = (
                    decoded_body.get("request_type")
                    if isinstance(decoded_body, Mapping)
                    else None
                )
                status, payload = route_value[discriminator]
            else:
                status, payload = route_value
            response_body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header(
                "X-DataRobot-Request-ID",
                f"surface-request-{len(type(self).requests)}",
            )
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), SurfaceHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, SurfaceHandler


def _server_base_url(server: ThreadingHTTPServer) -> str:
    host = server.server_address[0]
    port = server.server_address[1]
    return f"http://{host}:{port}"