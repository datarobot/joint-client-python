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

"""Maximum-utilization check against a live JointFM deployment.

The check reads the capacity envelope a deployment advertises in `/healthz`
and sizes the largest sample forecast that envelope admits: ``max_series``
target columns, an ``n_input``-row history, an ``n_output``-step horizon, and
the endpoint's full sample budget. Every column is a target, because targets
are what the response carries, so this is also the largest response the
deployment can be asked for. Every request slot the reachable endpoints
advertise (``max_concurrent_requests`` per endpoint) is filled at once with
one such request each, so every resource of every endpoint is in use. The
outcome is a pass or fail verdict plus the wall-clock time of that burst.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import time

from jointfm_client.capabilities import ForecastPlan, plan_forecast_columns
from jointfm_client.client import JointFMClient
from jointfm_client.contract import DataGenerationCapabilities
from jointfm_client.exceptions import (
    JointFMCapacityError,
    JointFMConfigurationError,
    JointFMError,
)

CONCURRENT_SATURATION_STAGE = "concurrent_saturation"
SYNTHETIC_COLUMN_PREFIX = "series_"
SYNTHETIC_START_VALUE = 100.0
SYNTHETIC_STEP_SCALE = 1.0
DEFAULT_UTILIZATION_SEED = 7


@dataclass(frozen=True, slots=True)
class UtilizationStage:
    """Outcome of one maximal forecast stage.

    ``n_samples`` is the total sample count the stage asked for and
    ``request_count`` how many maximal requests it sent at once, one per
    request slot. ``wallclock_seconds`` spans request building, every round
    trip, and response parsing, so it bounds each single request's duration
    from above. ``errors`` holds one entry per request that ended in a
    JointFM error or a wrong sample count.
    """

    name: str
    n_samples: int
    request_count: int
    wallclock_seconds: float
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every request completed without error."""
        return not self.errors


@dataclass(frozen=True, slots=True)
class MaxUtilizationReport:
    """Verdict and timings of one maximum-utilization check.

    ``envelope`` is the advertised capacity the requests were sized from.
    ``unavailable_instances`` lists configured deployment IDs that failed the
    health probe; any entry fails the check, because the pool then runs below
    its configured capacity. ``request_slots`` is the sum of
    ``max_concurrent_requests`` over the reachable endpoints.
    """

    envelope: DataGenerationCapabilities
    decoding_strategy: str
    topology_label: str
    request_slots: int
    unavailable_instances: tuple[str, ...]
    read_timeout_seconds: float
    health_seconds: float
    concurrent_saturation: UtilizationStage

    @property
    def passed(self) -> bool:
        """Return whether every endpoint was reachable and the burst passed."""
        return not self.unavailable_instances and self.concurrent_saturation.passed

    def format_lines(self) -> list[str]:
        """Return a human-readable report ending in the verdict line."""
        envelope = self.envelope
        lines = [
            f"envelope:      max_series={envelope.max_series} "
            f"n_input={envelope.n_input} n_output={envelope.n_output}",
            f"decoding:      {self.decoding_strategy}",
            f"topology:      {self.topology_label}",
            f"request slots: {self.request_slots}",
            f"read timeout:  {self.read_timeout_seconds:.1f}s per request",
            f"health probe:  {self.health_seconds:.2f}s",
        ]
        lines.extend(
            f"unavailable:   {deployment_id}"
            for deployment_id in self.unavailable_instances
        )
        stage = self.concurrent_saturation
        status = "PASS" if stage.passed else "FAIL"
        lines.append(
            f"{stage.name}: {status} n_samples={stage.n_samples} "
            f"requests={stage.request_count} "
            f"wallclock={stage.wallclock_seconds:.2f}s"
        )
        lines.extend(f"  error: {error}" for error in stage.errors)
        lines.append(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return lines


@dataclass(frozen=True, slots=True)
class _MaximalRequest:
    """The envelope-sized forecast inputs every request slot sends."""

    history: Sequence[Mapping[str, float]]
    plan: ForecastPlan
    query_times: Sequence[int]

    def send(self, client: JointFMClient, *, n_samples: int, seed: int) -> str | None:
        """Send one forecast; return an error when the sample count is wrong.

        JointFM errors propagate to the caller, which records them.
        """
        result = client.forecast_samples(
            self.history,
            columns=self.plan.columns,
            query_times=self.query_times,
            requested_columns=self.plan.requested_columns,
            n_samples=n_samples,
            seed=seed,
        )
        if len(result.samples) != n_samples:
            return f"expected {n_samples} samples, got {len(result.samples)}"
        return None


@dataclass(frozen=True, slots=True)
class _RequestSlot:
    """One request the saturation stage sends at the same time as the others."""

    deployment_id: str | None
    n_samples: int
    seed: int


def run_max_utilization_check(
    client: JointFMClient,
    *,
    seed: int = DEFAULT_UTILIZATION_SEED,
) -> MaxUtilizationReport:
    """Fill every advertised request slot at once and report the verdict.

    ``concurrent_saturation`` fills every request slot of every reachable
    endpoint at once: each endpoint receives ``max_concurrent_requests``
    simultaneous maximal requests for its own ``max_sample_count``, each
    through its own single-endpoint client (see
    :meth:`JointFMClient.endpoint_client`), so neither the pool's sample
    split nor its per-endpoint lock reshapes or serializes them. Slot ``i``
    uses seed ``seed + i``.

    The client must be built with ``JointFMRetryConfig(max_attempts=1)``: a
    retried request would turn a failure at maximum load into a slower
    success.

    Raises :class:`JointFMConfigurationError` when the client retries and
    :class:`JointFMCapacityError` when the deployment advertises no
    ``data_generation`` block. Errors raised during the burst are recorded in
    the stage instead, so the report still shows them.
    """
    if client.retry_config.max_attempts != 1:
        raise JointFMConfigurationError(
            "run_max_utilization_check requires a client built with "
            "JointFMRetryConfig(max_attempts=1); got "
            f"max_attempts={client.retry_config.max_attempts}"
        )

    health_started = time.perf_counter()
    health = client.health(cache=True, refresh=True)
    instances = client.health_instances(cache=True)
    health_seconds = time.perf_counter() - health_started

    envelope = health.data_generation
    if envelope is None:
        raise JointFMCapacityError(
            "JointFM deployment health metadata is missing the 'data_generation' "
            "block; the maximum-utilization check has no envelope to size from."
        )
    plan = plan_forecast_columns(
        health=health,
        feature_columns=(),
        target_columns=[
            f"{SYNTHETIC_COLUMN_PREFIX}{index}" for index in range(envelope.max_series)
        ],
        history_length=envelope.n_input,
        query_times_length=envelope.n_output,
    )
    request = _MaximalRequest(
        history=_synthetic_history(plan.target_columns, envelope.n_input, seed=seed),
        plan=plan,
        query_times=list(range(envelope.n_input, envelope.n_input + envelope.n_output)),
    )
    endpoint_slots = [
        (instance.deployment_id, instance.metadata)
        for instance in instances.instances
        if instance.metadata is not None
        for _ in range(instance.metadata.max_concurrent_requests)
    ]
    slots = tuple(
        _RequestSlot(
            deployment_id=deployment_id,
            n_samples=metadata.max_sample_count,
            seed=seed + index,
        )
        for index, (deployment_id, metadata) in enumerate(endpoint_slots)
    )

    concurrent_saturation = _run_concurrent_saturation(
        client, request=request, slots=slots
    )

    return MaxUtilizationReport(
        envelope=envelope,
        decoding_strategy=health.decoding_strategy,
        topology_label=instances.topology_label,
        request_slots=len(slots),
        unavailable_instances=tuple(
            str(instance.deployment_id)
            for instance in instances.instances
            if instance.metadata is None
        ),
        read_timeout_seconds=client.timeout.read_seconds,
        health_seconds=health_seconds,
        concurrent_saturation=concurrent_saturation,
    )


def _run_concurrent_saturation(
    client: JointFMClient,
    *,
    request: _MaximalRequest,
    slots: Sequence[_RequestSlot],
) -> UtilizationStage:
    """Send one maximal request per slot, all at once, and record each outcome.

    Every slot client probes health before the burst starts, so the timed
    burst holds forecasts only; a failed probe ends the stage before any
    forecast is sent.
    """
    total_samples = sum(slot.n_samples for slot in slots)
    slot_clients = tuple(client.endpoint_client(slot.deployment_id) for slot in slots)
    try:
        for slot_client in slot_clients:
            slot_client.health(cache=True)
    # A service or transport failure under maximum load is the verdict itself,
    # so it is recorded rather than propagated; other errors still raise.
    except JointFMError as caught:
        return UtilizationStage(
            name=CONCURRENT_SATURATION_STAGE,
            n_samples=total_samples,
            request_count=len(slots),
            wallclock_seconds=0.0,
            errors=(
                f"health probe before the burst: {type(caught).__name__}: {caught}",
            ),
        )

    def _send_slot(index: int) -> str | None:
        slot = slots[index]
        try:
            error = request.send(
                slot_clients[index], n_samples=slot.n_samples, seed=slot.seed
            )
        # Recorded per slot for the same reason as a failed health probe.
        except JointFMError as caught:
            error = f"{type(caught).__name__}: {caught}"
        return (
            None if error is None else f"slot {index} ({slot.deployment_id}): {error}"
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        outcomes = tuple(executor.map(_send_slot, range(len(slots))))
    wallclock_seconds = time.perf_counter() - started
    return UtilizationStage(
        name=CONCURRENT_SATURATION_STAGE,
        n_samples=total_samples,
        request_count=len(slots),
        wallclock_seconds=wallclock_seconds,
        errors=tuple(error for error in outcomes if error is not None),
    )


def _synthetic_history(
    column_names: Sequence[str],
    row_count: int,
    *,
    seed: int,
) -> list[dict[str, float]]:
    """Build seeded random-walk history rows, one walk per column."""
    generator = random.Random(seed)
    levels = dict.fromkeys(column_names, SYNTHETIC_START_VALUE)
    rows: list[dict[str, float]] = []
    for _ in range(row_count):
        for name in column_names:
            levels[name] += generator.gauss(0.0, SYNTHETIC_STEP_SCALE)
        rows.append(dict(levels))
    return rows


__all__ = [
    "DEFAULT_UTILIZATION_SEED",
    "MaxUtilizationReport",
    "UtilizationStage",
    "run_max_utilization_check",
]
