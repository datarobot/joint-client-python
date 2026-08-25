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

"""Round-robin load balancing across multiple hosted JointFM deployments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import logging
import threading
from typing import Any

from jointfm_client.configuration import DEFAULT_RETRY_STATUS_CODES
from jointfm_client.contract import (
    HEALTH_REQUEST_TYPE,
    HealthMetadata,
    validate_service_metadata,
)
from jointfm_client.exceptions import (
    JointFMHTTPStatusError,
    JointFMRequestError,
    UnsupportedModelVersionError,
    UnsupportedServiceContractError,
)
from jointfm_client.settings import JointFMInstanceSettings
from jointfm_client.transport import JSONTransport

logger = logging.getLogger(__name__)

_POOL_RETRYABLE_HTTP_STATUS_CODES = frozenset(DEFAULT_RETRY_STATUS_CODES)


class JointFMInstancePool:
    """Round-robin JointFM requests across hosted deployments.

    Use a fail-fast transport (``max_attempts=1``); this pool retries peers.
    """

    def __init__(
        self,
        *,
        instances: Sequence[JointFMInstanceSettings],
        transport: JSONTransport,
        expected_model_version: str | None = None,
    ) -> None:
        if len(instances) < 2:
            raise ValueError("JointFMInstancePool requires at least two instances")
        self._instances = tuple(instances)
        self._transport = transport
        self._expected_model_version = expected_model_version
        self._lock = threading.Lock()
        self._index = 0

    def next_instance(self) -> JointFMInstanceSettings:
        """Return the next instance using round-robin selection."""
        with self._lock:
            instance = self._instances[self._index]
            self._index = (self._index + 1) % len(self._instances)
            return instance

    def probe_all_health(self) -> HealthMetadata:
        """Probe peers; require matching model/checkpoint; return min sample cap."""
        healthy: list[tuple[JointFMInstanceSettings, HealthMetadata]] = []
        last_error: BaseException | None = None
        for instance in self._instances:
            try:
                payload = self._transport.post_json(
                    instance.predict_url, {"request_type": HEALTH_REQUEST_TYPE}
                )
                validate_service_metadata(
                    payload, expected_model_version=self._expected_model_version
                )
                healthy.append((instance, HealthMetadata.from_payload(payload)))
            except Exception as error:
                if not _is_pool_retryable(error):
                    raise
                last_error = error
                _log_unavailable(instance.deployment_id, error)

        if not healthy:
            assert last_error is not None
            raise last_error

        reference = healthy[0][1]
        max_samples = reference.max_sample_count
        for instance, metadata in healthy[1:]:
            if metadata.model_version != reference.model_version:
                raise UnsupportedModelVersionError(
                    "JointFM deployment pool model_version mismatch: "
                    f"{instance.deployment_id!r} advertises {metadata.model_version!r}, "
                    f"expected {reference.model_version!r}"
                )
            if metadata.checkpoint_version != reference.checkpoint_version:
                raise UnsupportedServiceContractError(
                    "JointFM deployment pool checkpoint_version mismatch: "
                    f"{instance.deployment_id!r} advertises "
                    f"{metadata.checkpoint_version!r}, "
                    f"expected {reference.checkpoint_version!r}"
                )
            max_samples = min(max_samples, metadata.max_sample_count)
        if max_samples == reference.max_sample_count:
            return reference
        return replace(reference, max_sample_count=max_samples)

    def post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST payload, trying untried peers on retryable failures."""
        tried: set[str] = set()
        last_error: BaseException | None = None
        while len(tried) < len(self._instances):
            instance = self._next_untried(tried)
            tried.add(instance.deployment_id)
            try:
                return self._transport.post_json(instance.predict_url, payload)
            except Exception as error:
                if not _is_pool_retryable(error):
                    raise
                last_error = error
                _log_unavailable(instance.deployment_id, error)
        assert last_error is not None
        raise last_error

    def _next_untried(self, tried: set[str]) -> JointFMInstanceSettings:
        for _ in range(len(self._instances)):
            instance = self.next_instance()
            if instance.deployment_id not in tried:
                return instance
        raise RuntimeError("JointFMInstancePool has no untried instances")


def _log_unavailable(deployment_id: str, error: BaseException) -> None:
    logger.warning(
        "JointFM instance unavailable: deployment_id=%s error=%s", deployment_id, error
    )


def _is_pool_retryable(error: BaseException) -> bool:
    if isinstance(error, JointFMRequestError):
        return True
    return (
        isinstance(error, JointFMHTTPStatusError)
        and error.status_code in _POOL_RETRYABLE_HTTP_STATUS_CODES
    )
