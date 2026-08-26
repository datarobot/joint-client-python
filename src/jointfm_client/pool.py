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
from dataclasses import dataclass, replace
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


class PoolPeer:
    """One hosted deployment endpoint with a Session-safe POST lock."""

    def __init__(
        self,
        settings: JointFMInstanceSettings,
        transport: JSONTransport,
        post_lock: threading.Lock,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self._post_lock = post_lock

    @property
    def deployment_id(self) -> str:
        """Deployment identifier used for routing and health accounting."""
        return self.settings.deployment_id

    def post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST ``payload`` to this peer's predict URL under the peer lock."""
        with self._post_lock:
            return self.transport.post_json(self.settings.predict_url, payload)


@dataclass(frozen=True, slots=True)
class HealthProbeResult:
    """Merged health metadata and the peer IDs that passed the gate."""

    metadata: HealthMetadata
    healthy_ids: frozenset[str]


class PeerRoutingState:
    """Active-set, cooldown, and round-robin cursor for pool peers."""

    def __init__(
        self,
        *,
        deployment_ids: Sequence[str],
        peer_cooldown_seconds: float,
    ) -> None:
        self._peer_cooldown_seconds = peer_cooldown_seconds
        self._lock = threading.Lock()
        self._index = 0
        self._active_ids = set(deployment_ids)
        self._cooldown_until: dict[str, float] = {}

    def eligible(self, peers: Sequence[PoolPeer]) -> tuple[PoolPeer, ...]:
        """Return peers currently preferred for routing."""
        with self._lock:
            return self._eligible_unlocked(peers)

    def next(self, peers: Sequence[PoolPeer]) -> PoolPeer:
        """Return the next eligible peer using round-robin selection."""
        with self._lock:
            active = self._eligible_unlocked(peers)
            peer = active[self._index % len(active)]
            self._index = (self._index + 1) % len(active)
            return peer

    def at(self, peers: Sequence[PoolPeer], index: int) -> PoolPeer:
        """Return the eligible peer pinned for ``index`` (sticky batch mapping)."""
        active = self.eligible(peers)
        return active[index % len(active)]

    def set_healthy(self, healthy_ids: Sequence[str]) -> None:
        """Replace the active set with health-passing peers and clear their cooldowns."""
        with self._lock:
            self._active_ids = set(healthy_ids)
            self._index = 0
            for deployment_id in healthy_ids:
                self._cooldown_until.pop(deployment_id, None)

    def cool_down(self, deployment_id: str) -> None:
        """Temporarily exclude ``deployment_id`` from preferred routing."""
        with self._lock:
            self._cooldown_until[deployment_id] = (
                time.monotonic() + self._peer_cooldown_seconds
            )

    def reactivate(self, deployment_id: str) -> None:
        """Mark ``deployment_id`` active and clear any cooldown."""
        with self._lock:
            self._active_ids.add(deployment_id)
            self._cooldown_until.pop(deployment_id, None)

    def _eligible_unlocked(self, peers: Sequence[PoolPeer]) -> tuple[PoolPeer, ...]:
        now = time.monotonic()
        active = tuple(peer for peer in peers if peer.deployment_id in self._active_ids)
        not_cooling = tuple(
            peer
            for peer in active
            if self._cooldown_until.get(peer.deployment_id, 0.0) <= now
        )
        # All cooling: still try health-active peers rather than stall.
        return not_cooling or active or tuple(peers)


class PoolHealthGate:
    """Probe peers and require matching model/checkpoint across healthy ones."""

    def __init__(self, *, expected_model_version: str | None = None) -> None:
        self._expected_model_version = expected_model_version

    def probe(self, peers: Sequence[PoolPeer]) -> HealthProbeResult:
        """Probe peers; skip per-peer failures; fail only when none are usable."""
        healthy: list[tuple[PoolPeer, HealthMetadata]] = []
        last_error: BaseException | None = None
        for peer in peers:
            try:
                payload = peer.post_json({"request_type": HEALTH_REQUEST_TYPE})
                validate_service_metadata(
                    payload, expected_model_version=self._expected_model_version
                )
                healthy.append((peer, HealthMetadata.from_payload(payload)))
            except Exception as error:
                # Skip unreachable or incompatible peers; a bad backup must not
                # take down a healthy primary.
                last_error = error
                _log_unavailable(peer.deployment_id, error)

        if not healthy:
            assert last_error is not None
            raise last_error

        reference = healthy[0][1]
        max_samples = reference.max_sample_count
        for peer, metadata in healthy[1:]:
            _reject_metadata_mismatch(peer.deployment_id, metadata, reference)
            max_samples = min(max_samples, metadata.max_sample_count)

        metadata = (
            reference
            if max_samples == reference.max_sample_count
            else replace(reference, max_sample_count=max_samples)
        )
        return HealthProbeResult(
            metadata=metadata,
            healthy_ids=frozenset(peer.deployment_id for peer, _ in healthy),
        )


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
        # Key locks by transport identity so a shared injected Session serializes,
        # while distinct per-peer Sessions stay concurrent.
        locks_by_transport: dict[int, threading.Lock] = {}
        peers: list[PoolPeer] = []
        for instance, transport in zip(instances, transports, strict=True):
            post_lock = locks_by_transport.setdefault(id(transport), threading.Lock())
            peers.append(PoolPeer(instance, transport, post_lock))
        self._peers = tuple(peers)
        self._peers_by_id = {peer.deployment_id: peer for peer in self._peers}
        self._routing = PeerRoutingState(
            deployment_ids=tuple(peer.deployment_id for peer in self._peers),
            peer_cooldown_seconds=peer_cooldown_seconds,
        )
        self._health_gate = PoolHealthGate(
            expected_model_version=expected_model_version
        )

    @property
    def instance_count(self) -> int:
        """Number of peers currently eligible for routing."""
        return len(self._routing.eligible(self._peers))

    def next_instance(self) -> JointFMInstanceSettings:
        """Return the next eligible instance using round-robin selection."""
        return self._routing.next(self._peers).settings

    def instance_at(self, index: int) -> JointFMInstanceSettings:
        """Return the eligible instance pinned for ``index`` (sticky batch mapping)."""
        return self._routing.at(self._peers, index).settings

    def probe_all_health(self) -> HealthMetadata:
        """Probe peers; require matching model/checkpoint; return min sample cap.

        Per-peer transport or contract failures skip that peer. The pool fails
        only when no peer is usable, or when usable peers disagree with each
        other on model/checkpoint.
        """
        result = self._health_gate.probe(self._peers)
        self._routing.set_healthy(tuple(result.healthy_ids))
        return result.metadata

    def post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST payload via round-robin, trying other peers on retryable failures."""
        return self.post_json_to(self.next_instance(), payload)

    def post_json_to(
        self, instance: JointFMInstanceSettings, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """POST to ``instance`` first; on retryable failure try remaining peers."""
        preferred = self._peers_by_id[instance.deployment_id]
        candidates = self._failover_candidates(preferred)
        last_error: BaseException | None = None
        for candidate in candidates:
            try:
                result = candidate.post_json(payload)
            except Exception as error:
                if not _is_pool_retryable(error):
                    raise
                last_error = error
                self._routing.cool_down(candidate.deployment_id)
                _log_unavailable(candidate.deployment_id, error)
                continue
            self._routing.reactivate(candidate.deployment_id)
            return result
        assert last_error is not None
        raise last_error

    def _failover_candidates(self, preferred: PoolPeer) -> tuple[PoolPeer, ...]:
        """Prefer health-eligible peers; then try health-excluded peers last."""
        eligible = self._routing.eligible(self._peers)
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
            peer for peer in self._peers if peer.deployment_id not in seen
        )
        return preferred_first + last_resort


def _reject_metadata_mismatch(
    deployment_id: str,
    metadata: HealthMetadata,
    reference: HealthMetadata,
) -> None:
    if metadata.model_version != reference.model_version:
        raise UnsupportedModelVersionError(
            "JointFM deployment pool model_version mismatch: "
            f"{deployment_id!r} advertises {metadata.model_version!r}, "
            f"expected {reference.model_version!r}"
        )
    if metadata.checkpoint_version != reference.checkpoint_version:
        raise UnsupportedServiceContractError(
            "JointFM deployment pool checkpoint_version mismatch: "
            f"{deployment_id!r} advertises {metadata.checkpoint_version!r}, "
            f"expected {reference.checkpoint_version!r}"
        )


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
