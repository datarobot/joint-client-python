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
import time
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
_DEFAULT_PEER_COOLDOWN_SECONDS = 30.0


class JointFMInstancePool:
    """Round-robin JointFM requests across hosted deployments.

    Use one fail-fast transport per peer (``max_attempts=1``); this pool retries
    peers. Each transport is locked independently so parallel batches stay
    concurrent across peers while failover onto a busy peer stays Session-safe.
    """

    def __init__(
        self,
        *,
        instances: Sequence[JointFMInstanceSettings],
        transports: Sequence[JSONTransport],
        expected_model_version: str | None = None,
        peer_cooldown_seconds: float = _DEFAULT_PEER_COOLDOWN_SECONDS,
    ) -> None:
        if len(instances) < 2:
            raise ValueError("JointFMInstancePool requires at least two instances")
        if len(transports) != len(instances):
            raise ValueError(
                "JointFMInstancePool requires one transport per instance: "
                f"got {len(transports)} transports for {len(instances)} instances"
            )
        if peer_cooldown_seconds < 0:
            raise ValueError("peer_cooldown_seconds must be >= 0")
        self._instances = tuple(instances)
        self._transports = {
            instance.deployment_id: transport
            for instance, transport in zip(instances, transports, strict=True)
        }
        # Key locks by transport identity so a shared injected Session serializes,
        # while distinct per-peer Sessions stay concurrent.
        self._post_locks = {
            id(transport): threading.Lock() for transport in self._transports.values()
        }
        self._expected_model_version = expected_model_version
        self._peer_cooldown_seconds = peer_cooldown_seconds
        self._lock = threading.Lock()
        self._index = 0
        self._active_ids = {instance.deployment_id for instance in self._instances}
        self._cooldown_until: dict[str, float] = {}

    @property
    def instance_count(self) -> int:
        """Number of peers currently eligible for routing."""
        return len(self._eligible_instances())

    def next_instance(self) -> JointFMInstanceSettings:
        """Return the next eligible instance using round-robin selection."""
        with self._lock:
            active = self._eligible_instances_unlocked()
            instance = active[self._index % len(active)]
            self._index = (self._index + 1) % len(active)
            return instance

    def instance_at(self, index: int) -> JointFMInstanceSettings:
        """Return the eligible instance pinned for ``index`` (sticky batch mapping)."""
        active = self._eligible_instances()
        return active[index % len(active)]

    def probe_all_health(self) -> HealthMetadata:
        """Probe peers; require matching model/checkpoint; return min sample cap.

        Per-peer transport or contract failures skip that peer. The pool fails
        only when no peer is usable, or when usable peers disagree with each
        other on model/checkpoint.
        """
        healthy: list[tuple[JointFMInstanceSettings, HealthMetadata]] = []
        last_error: BaseException | None = None
        for instance in self._instances:
            try:
                payload = self._post_json(
                    instance, {"request_type": HEALTH_REQUEST_TYPE}
                )
                validate_service_metadata(
                    payload, expected_model_version=self._expected_model_version
                )
                healthy.append((instance, HealthMetadata.from_payload(payload)))
            except Exception as error:
                # Skip unreachable or incompatible peers; a bad backup must not
                # take down a healthy primary.
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

        with self._lock:
            self._active_ids = {instance.deployment_id for instance, _ in healthy}
            self._index = 0
            for instance, _ in healthy:
                self._cooldown_until.pop(instance.deployment_id, None)

        if max_samples == reference.max_sample_count:
            return reference
        return replace(reference, max_sample_count=max_samples)

    def post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST payload via round-robin, trying other peers on retryable failures."""
        return self.post_json_to(self.next_instance(), payload)

    def post_json_to(
        self, instance: JointFMInstanceSettings, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """POST to ``instance`` first; on retryable failure try remaining peers."""
        candidates = self._failover_candidates(instance)
        last_error: BaseException | None = None
        for candidate in candidates:
            try:
                result = self._post_json(candidate, payload)
            except Exception as error:
                if not _is_pool_retryable(error):
                    raise
                last_error = error
                self._cool_down(candidate.deployment_id)
                _log_unavailable(candidate.deployment_id, error)
                continue
            self._reactivate(candidate.deployment_id)
            return result
        assert last_error is not None
        raise last_error

    def _failover_candidates(
        self, preferred: JointFMInstanceSettings
    ) -> tuple[JointFMInstanceSettings, ...]:
        """Prefer health-eligible peers; then try health-excluded peers last."""
        eligible = self._eligible_instances()
        if preferred.deployment_id in {peer.deployment_id for peer in eligible}:
            preferred_first = (preferred,) + tuple(
                peer
                for peer in eligible
                if peer.deployment_id != preferred.deployment_id
            )
        else:
            preferred_first = eligible
        seen = {peer.deployment_id for peer in preferred_first}
        last_resort = tuple(
            peer for peer in self._instances if peer.deployment_id not in seen
        )
        return preferred_first + last_resort

    def _eligible_instances(self) -> tuple[JointFMInstanceSettings, ...]:
        with self._lock:
            return self._eligible_instances_unlocked()

    def _eligible_instances_unlocked(self) -> tuple[JointFMInstanceSettings, ...]:
        now = time.monotonic()
        active = tuple(
            instance
            for instance in self._instances
            if instance.deployment_id in self._active_ids
        )
        not_cooling = tuple(
            instance
            for instance in active
            if self._cooldown_until.get(instance.deployment_id, 0.0) <= now
        )
        # All cooling: still try health-active peers rather than stall.
        return not_cooling or active or self._instances

    def _cool_down(self, deployment_id: str) -> None:
        with self._lock:
            self._cooldown_until[deployment_id] = (
                time.monotonic() + self._peer_cooldown_seconds
            )

    def _reactivate(self, deployment_id: str) -> None:
        with self._lock:
            self._active_ids.add(deployment_id)
            self._cooldown_until.pop(deployment_id, None)

    def _post_json(
        self, instance: JointFMInstanceSettings, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        transport = self._transports[instance.deployment_id]
        with self._post_locks[id(transport)]:
            return transport.post_json(instance.predict_url, payload)


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
