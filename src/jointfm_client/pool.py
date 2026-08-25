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
import logging
import threading
from typing import Any

from jointfm_client.contract import (
    HEALTH_REQUEST_TYPE,
    HealthMetadata,
    validate_service_metadata,
)
from jointfm_client.exceptions import (
    JointFMHTTPStatusError,
    JointFMRequestError,
    UnsupportedModelVersionError,
)
from jointfm_client.settings import JointFMInstanceSettings
from jointfm_client.transport import JSONTransport

logger = logging.getLogger(__name__)

_POOL_RETRYABLE_HTTP_STATUS_CODES = frozenset({502, 503, 504})


class JointFMInstancePool:
    """Distributes JointFM requests across multiple hosted deployment instances."""

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
        """Probe instances; log unavailable ones and require matching model_version."""
        metadata_by_instance: list[tuple[JointFMInstanceSettings, HealthMetadata]] = []
        last_error: BaseException | None = None
        for instance in self._instances:
            try:
                payload = self._fetch_hosted_health_payload(instance)
                validate_service_metadata(
                    payload,
                    expected_model_version=self._expected_model_version,
                )
                metadata_by_instance.append(
                    (instance, HealthMetadata.from_payload(payload))
                )
            except Exception as error:
                if not _should_retry_on_next_instance(error):
                    raise
                last_error = error
                logger.warning(
                    "JointFM instance unavailable: deployment_id=%s error=%s",
                    instance.deployment_id,
                    error,
                )

        if not metadata_by_instance:
            assert last_error is not None
            raise last_error

        reference = metadata_by_instance[0][1]
        for instance, metadata in metadata_by_instance[1:]:
            if metadata.model_version != reference.model_version:
                raise UnsupportedModelVersionError(
                    "JointFM deployment pool model_version mismatch: "
                    f"{instance.deployment_id!r} advertises {metadata.model_version!r}, "
                    f"expected {reference.model_version!r}"
                )
        return reference

    def post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST one payload, trying untried instances on retryable failures."""
        tried: set[str] = set()
        last_error: BaseException | None = None
        while len(tried) < len(self._instances):
            instance = self._next_untried_instance(tried)
            tried.add(instance.deployment_id)
            try:
                return self._transport.post_json(instance.predict_url, payload)
            except Exception as error:
                if not _should_retry_on_next_instance(error):
                    raise
                last_error = error
                logger.warning(
                    "JointFM instance unavailable: deployment_id=%s error=%s",
                    instance.deployment_id,
                    error,
                )
        assert last_error is not None
        raise last_error

    def _next_untried_instance(self, tried: set[str]) -> JointFMInstanceSettings:
        for _ in range(len(self._instances)):
            instance = self.next_instance()
            if instance.deployment_id not in tried:
                return instance
        raise RuntimeError("JointFMInstancePool has no untried instances")

    def _fetch_hosted_health_payload(
        self,
        instance: JointFMInstanceSettings,
    ) -> Mapping[str, Any]:
        return self._transport.post_json(
            instance.predict_url,
            {"request_type": HEALTH_REQUEST_TYPE},
        )


def _should_retry_on_next_instance(error: BaseException) -> bool:
    if isinstance(error, JointFMRequestError):
        return True
    return (
        isinstance(error, JointFMHTTPStatusError)
        and error.status_code in _POOL_RETRYABLE_HTTP_STATUS_CODES
    )
