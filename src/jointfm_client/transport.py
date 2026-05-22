"""Synchronous JSON transport for JointFM HTTP calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Final, Protocol

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from jointfm_client.configuration import (
    DATAROBOT_REQUEST_ID_HEADERS,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS,
    DEFAULT_RETRYABLE_METHODS,
    DEFAULT_RETRY_STATUS_CODES,
    USER_AGENT_HEADER,
)
from jointfm_client.contract import DISTRIBUTION_NAME, PACKAGE_VERSION
from jointfm_client.exceptions import (
    JointFMConfigurationError,
    JointFMHTTPStatusError,
    JointFMRequestEncodingError,
    JointFMRequestError,
    JointFMResponseDecodeError,
)
from jointfm_client.settings import JointFMSettings, build_datarobot_prediction_headers

RETRYABLE_METHODS: Final = frozenset(DEFAULT_RETRYABLE_METHODS)


def _require_positive_finite(value: float, field: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise JointFMConfigurationError(f"{field} must be finite and positive")


class JSONTransport(Protocol):
    """Minimal JSON transport interface used by the public client."""

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Return one decoded JSON object from an HTTP GET request."""

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one decoded JSON object from an HTTP POST request."""


@dataclass(frozen=True, slots=True)
class JointFMTimeoutConfig:
    """Connect and read timeout settings for one HTTP request."""

    connect_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Validate that both timeout phases are finite and positive."""
        _require_positive_finite(self.connect_seconds, "connect_seconds")
        _require_positive_finite(self.read_seconds, "read_seconds")

    def as_requests_timeout(self) -> tuple[float, float]:
        """Return the tuple shape expected by `requests`."""
        return (self.connect_seconds, self.read_seconds)


@dataclass(frozen=True, slots=True)
class JointFMRetryConfig:
    """Retry settings for transient network and server failures."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    status_codes: Sequence[int] = field(default_factory=lambda: DEFAULT_RETRY_STATUS_CODES)
    allowed_methods: Sequence[str] = field(default_factory=lambda: DEFAULT_RETRYABLE_METHODS)

    def __post_init__(self) -> None:
        """Validate retry limits and retryable response codes."""
        if self.max_attempts < 1:
            raise JointFMConfigurationError("max_attempts must be at least 1")
        if not math.isfinite(self.backoff_seconds) or self.backoff_seconds < 0:
            raise JointFMConfigurationError("backoff_seconds must be finite and non-negative")
        for status_code in self.status_codes:
            if status_code < 400:
                raise JointFMConfigurationError("retry status_codes must be HTTP error statuses")
        if isinstance(self.allowed_methods, str | bytes | bytearray):
            raise JointFMConfigurationError("allowed_methods must be a sequence of HTTP methods")
        for method in self.allowed_methods:
            if method == "" or method.strip() != method:
                raise JointFMConfigurationError("allowed_methods must be non-empty HTTP methods")

    def as_urllib3_retry(self) -> Retry:
        """Return the retry policy object consumed by `requests` adapters."""
        retry_count = self.max_attempts - 1
        return Retry(
            total=retry_count,
            connect=retry_count,
            read=retry_count,
            status=retry_count,
            backoff_factor=self.backoff_seconds,
            status_forcelist=tuple(self.status_codes),
            allowed_methods=frozenset(self.allowed_methods),
            raise_on_status=False,
            respect_retry_after_header=True,
        )


class JointFMHTTPTransport:
    """Small `requests.Session` wrapper for JointFM JSON endpoints."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: JointFMTimeoutConfig = JointFMTimeoutConfig(),
        retry_config: JointFMRetryConfig = JointFMRetryConfig(),
        user_agent: str | None = None,
        response_body_excerpt_characters: int = DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS,
        datarobot_request_id_headers: Sequence[str] = DATAROBOT_REQUEST_ID_HEADERS,
    ) -> None:
        """Configure one session with headers, timeouts, retries, and user-agent."""
        self._session = session or requests.Session()
        self._headers = _headers_with_user_agent(headers or {}, user_agent)
        self._timeout = timeout
        self._response_body_excerpt_characters = _require_positive_integer(
            response_body_excerpt_characters,
            "response_body_excerpt_characters",
        )
        self._datarobot_request_id_headers = _require_non_empty_string_sequence(
            datarobot_request_id_headers,
            "datarobot_request_id_headers",
        )
        adapter = HTTPAdapter(max_retries=retry_config.as_urllib3_retry())
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @classmethod
    def from_settings(
        cls,
        settings: JointFMSettings,
        *,
        session: requests.Session | None = None,
        timeout: JointFMTimeoutConfig = JointFMTimeoutConfig(),
        retry_config: JointFMRetryConfig = JointFMRetryConfig(),
        user_agent: str | None = None,
        response_body_excerpt_characters: int = DEFAULT_RESPONSE_BODY_EXCERPT_CHARACTERS,
        datarobot_request_id_headers: Sequence[str] = DATAROBOT_REQUEST_ID_HEADERS,
    ) -> "JointFMHTTPTransport":
        """Create an authenticated hosted DataRobot transport from SDK settings."""
        return cls(
            session=session,
            headers=build_datarobot_prediction_headers(settings.datarobot_api_token),
            timeout=timeout,
            retry_config=retry_config,
            user_agent=user_agent,
            response_body_excerpt_characters=response_body_excerpt_characters,
            datarobot_request_id_headers=datarobot_request_id_headers,
        )

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Return one decoded JSON object from an HTTP GET request."""
        return self._request_json("GET", url)

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one decoded JSON object from an HTTP POST request."""
        return self._request_json("POST", url, payload=payload)

    def close(self) -> None:
        """Close the underlying session and pooled connections."""
        self._session.close()

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        try:
            response = self._session.request(
                method,
                url,
                headers=self._headers,
                json=payload,
                timeout=self._timeout.as_requests_timeout(),
            )
        except TypeError as error:
            raise JointFMRequestEncodingError(
                "JointFM request payload must be JSON-serializable"
            ) from error
        except requests.RequestException as error:
            raise JointFMRequestError(f"JointFM HTTP request failed: {error}") from error

        response_payload = _decode_json_object(
            response,
            response_body_excerpt_characters=self._response_body_excerpt_characters,
            datarobot_request_id_headers=self._datarobot_request_id_headers,
        )
        if response.status_code >= 400:
            raise _http_status_error(
                response,
                response_payload,
                response_body_excerpt_characters=self._response_body_excerpt_characters,
                datarobot_request_id_headers=self._datarobot_request_id_headers,
            )
        return response_payload


def _headers_with_user_agent(
    headers: Mapping[str, str],
    user_agent: str | None,
) -> dict[str, str]:
    merged_headers = dict(headers)
    if USER_AGENT_HEADER not in merged_headers:
        merged_headers[USER_AGENT_HEADER] = user_agent or _default_user_agent()
    return merged_headers


def _default_user_agent() -> str:
    return f"{DISTRIBUTION_NAME}/{PACKAGE_VERSION}"


def _decode_json_object(
    response: requests.Response,
    *,
    response_body_excerpt_characters: int,
    datarobot_request_id_headers: Sequence[str],
) -> Mapping[str, Any]:
    body_excerpt = _response_body_excerpt(response, response_body_excerpt_characters)
    datarobot_request_id = _datarobot_request_id(response, datarobot_request_id_headers)
    if body_excerpt == "":
        raise JointFMResponseDecodeError(
            "JointFM service returned an empty response body",
            status_code=_response_status_code(response),
            response_body_excerpt=body_excerpt,
            datarobot_request_id=datarobot_request_id,
        )

    try:
        response_payload = response.json()
    except ValueError as error:
        raise JointFMResponseDecodeError(
            "JointFM service returned a non-JSON response body",
            status_code=_response_status_code(response),
            response_body_excerpt=body_excerpt,
            datarobot_request_id=datarobot_request_id,
        ) from error

    if not isinstance(response_payload, Mapping):
        raise JointFMResponseDecodeError(
            "JointFM service returned a JSON response that is not an object",
            status_code=_response_status_code(response),
            response_body_excerpt=body_excerpt,
            datarobot_request_id=datarobot_request_id,
        )
    return response_payload


def _http_status_error(
    response: requests.Response,
    response_payload: Mapping[str, Any],
    *,
    response_body_excerpt_characters: int,
    datarobot_request_id_headers: Sequence[str],
) -> JointFMHTTPStatusError:
    jointfm_errors = _jointfm_errors(response_payload)
    first_error_message = _first_jointfm_error_string(jointfm_errors, "message")
    status_code = _response_status_code(response)
    message = f"JointFM service returned HTTP {status_code}"
    if first_error_message is not None:
        message = f"{message}: {first_error_message}"
    return JointFMHTTPStatusError(
        message,
        status_code=status_code,
        response_body_excerpt=_response_body_excerpt(
            response,
            response_body_excerpt_characters,
        ),
        datarobot_request_id=_datarobot_request_id(response, datarobot_request_id_headers),
        jointfm_errors=jointfm_errors,
    )


def _jointfm_errors(response_payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    errors = response_payload.get("errors")
    if not isinstance(errors, Sequence) or isinstance(errors, str | bytes | bytearray):
        return ()

    parsed_errors: list[Mapping[str, Any]] = []
    for error in errors:
        if isinstance(error, Mapping):
            parsed_errors.append(dict(error))
    return tuple(parsed_errors)


def _first_jointfm_error_string(
    errors: Sequence[Mapping[str, Any]],
    field: str,
) -> str | None:
    if not errors:
        return None
    value = errors[0].get(field)
    if isinstance(value, str) and value != "":
        return value
    return None


def _datarobot_request_id(
    response: requests.Response,
    datarobot_request_id_headers: Sequence[str],
) -> str | None:
    for header in datarobot_request_id_headers:
        value = response.headers.get(header)
        if value is not None and value != "":
            return value
    return None


def _response_body_excerpt(response: requests.Response, character_count: int) -> str:
    return response.text.strip()[:character_count]


def _response_status_code(response: requests.Response) -> int:
    status_code = response.status_code
    if not isinstance(status_code, int):
        raise JointFMRequestError("JointFM HTTP response did not include a status code")
    return status_code


def _require_positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JointFMConfigurationError(f"{field} must be a positive integer")
    return value


def _require_non_empty_string_sequence(
    values: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
        raise JointFMConfigurationError(f"{field} must be a sequence of strings")
    tuple_values = tuple(values)
    if not tuple_values:
        raise JointFMConfigurationError(f"{field} must not be empty")
    for value in tuple_values:
        if value == "" or value.strip() != value:
            raise JointFMConfigurationError(f"{field} entries must be non-empty")
    return tuple_values