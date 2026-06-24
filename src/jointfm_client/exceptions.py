"""SDK exception hierarchy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class JointFMError(Exception):
    """Base class for JointFM SDK errors."""


class JointFMServiceError(JointFMError):
    """Raised when a successful HTTP response still carries JointFM service errors."""

    def __init__(
        self,
        message: str,
        *,
        jointfm_errors: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Store structured JointFM errors returned in a JSON response body."""
        super().__init__(message)
        self.jointfm_errors = tuple(jointfm_errors)


class JointFMConfigurationError(JointFMError):
    """Raised when local settings or deployment selection are invalid."""


class JointFMCapacityError(JointFMConfigurationError):
    """Raised when a planned forecast exceeds the deployed model's capacity envelope."""


class JointFMTransportError(JointFMError):
    """Base class for HTTP transport and response errors."""


class JointFMRequestEncodingError(JointFMTransportError):
    """Raised when an outbound JSON request cannot be encoded."""


class JointFMRequestError(JointFMTransportError):
    """Raised when a network request fails before a usable response is received."""


class JointFMResponseError(JointFMTransportError):
    """Raised when a service response is malformed or unsuccessful."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body_excerpt: str,
        datarobot_request_id: str | None = None,
        jointfm_errors: Sequence[Mapping[str, Any]] = (),
        retry_after_seconds: float | None = None,
    ) -> None:
        """Store HTTP metadata from one malformed or unsuccessful response."""
        super().__init__(message)
        self.status_code = status_code
        self.response_body_excerpt = response_body_excerpt
        self.datarobot_request_id = datarobot_request_id
        self.jointfm_errors = tuple(jointfm_errors)
        self.retry_after_seconds = retry_after_seconds


class JointFMResponseDecodeError(JointFMResponseError):
    """Raised when a service response is empty, non-JSON, or not a JSON object."""


class JointFMHTTPStatusError(JointFMResponseError):
    """Raised when the service returns a non-success HTTP status."""


class JointFMCompatibilityError(JointFMError):
    """Base class for fail-fast service compatibility errors."""


class UnsupportedSchemaVersionError(JointFMCompatibilityError):
    """Raised when the service advertises a schema version unsupported by the SDK."""


class UnsupportedModelVersionError(JointFMCompatibilityError):
    """Raised when the service model version differs from the requested version."""


class UnsupportedServiceContractError(JointFMCompatibilityError):
    """Raised when advertised service capabilities do not match the V1 contract."""