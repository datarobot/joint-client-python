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
and sends the largest sample forecast that envelope admits: ``max_series``
target columns, an ``n_input``-row history, an ``n_output``-step horizon, and
the full sample budget. Every column is a target, because targets are what the
response carries, so this is also the largest response the deployment can be
asked for. A pool of deployments is additionally saturated with one request
per endpoint at once. The outcome is a pass or fail verdict plus the
wall-clock time of every stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
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

POOL_LOGGER_NAME = "jointfm_client.pool"
SINGLE_REQUEST_STAGE = "single_request"
POOL_SATURATION_STAGE = "pool_saturation"
SYNTHETIC_COLUMN_PREFIX = "series_"
SYNTHETIC_START_VALUE = 100.0
SYNTHETIC_STEP_SCALE = 1.0
DEFAULT_UTILIZATION_SEED = 7


@dataclass(frozen=True, slots=True)
class UtilizationStage:
    """Outcome of one maximal forecast stage.

    ``n_samples`` is the sample count the stage asked for and
    ``request_count`` how many HTTP forecast requests the client split it into:
    one per endpoint-sized sample batch. ``wallclock_seconds`` spans request
    building, every round trip, and response parsing, so it bounds each single
    request's duration from above. ``failovers`` holds the pool's
    "instance unavailable" warnings logged during the stage: the pool reroutes
    a failed request to another endpoint, so the forecast can still succeed
    although one endpoint could not serve it, and the stage counts that as a
    failure. ``error`` is the JointFM error that ended the stage, if any.
    """

    name: str
    n_samples: int
    request_count: int
    wallclock_seconds: float
    failovers: tuple[str, ...]
    error: str | None

    @property
    def passed(self) -> bool:
        """Return whether the stage completed without error or failover."""
        return self.error is None and not self.failovers


@dataclass(frozen=True, slots=True)
class MaxUtilizationReport:
    """Verdict and timings of one maximum-utilization check.

    ``envelope`` is the advertised capacity the requests were sized from.
    ``unavailable_instances`` lists configured deployment IDs that failed the
    health probe; any entry fails the check, because the pool then runs below
    its configured capacity. ``pool_saturation`` is ``None`` for a single
    endpoint, where it would repeat ``single_request``.
    """

    envelope: DataGenerationCapabilities
    decoding_strategy: str
    topology_label: str
    unavailable_instances: tuple[str, ...]
    read_timeout_seconds: float
    health_seconds: float
    single_request: UtilizationStage
    pool_saturation: UtilizationStage | None

    @property
    def stages(self) -> tuple[UtilizationStage, ...]:
        """Return the stages that ran, in execution order."""
        if self.pool_saturation is None:
            return (self.single_request,)
        return (self.single_request, self.pool_saturation)

    @property
    def passed(self) -> bool:
        """Return whether every endpoint was reachable and every stage passed."""
        return not self.unavailable_instances and all(
            stage.passed for stage in self.stages
        )

    def format_lines(self) -> list[str]:
        """Return a human-readable report ending in the verdict line."""
        envelope = self.envelope
        lines = [
            f"envelope:      max_series={envelope.max_series} "
            f"n_input={envelope.n_input} n_output={envelope.n_output}",
            f"decoding:      {self.decoding_strategy}",
            f"topology:      {self.topology_label}",
            f"read timeout:  {self.read_timeout_seconds:.1f}s per request",
            f"health probe:  {self.health_seconds:.2f}s",
        ]
        lines.extend(
            f"unavailable:   {deployment_id}"
            for deployment_id in self.unavailable_instances
        )
        if self.pool_saturation is None:
            lines.append(
                f"{POOL_SATURATION_STAGE}: not applicable (one endpoint configured)"
            )
        for stage in self.stages:
            status = "PASS" if stage.passed else "FAIL"
            lines.append(
                f"{stage.name}: {status} n_samples={stage.n_samples} "
                f"requests={stage.request_count} "
                f"wallclock={stage.wallclock_seconds:.2f}s"
            )
            lines.extend(f"  failover: {message}" for message in stage.failovers)
            if stage.error is not None:
                lines.append(f"  error: {stage.error}")
        lines.append(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return lines


def run_max_utilization_check(
    client: JointFMClient,
    *,
    seed: int = DEFAULT_UTILIZATION_SEED,
) -> MaxUtilizationReport:
    """Send the largest advertised sample forecast and report the verdict.

    The client must be built with ``JointFMRetryConfig(max_attempts=1)``: a
    retried request would turn a failure at maximum load into a slower
    success. Pool endpoints already send each request once and fail over
    instead, and those failovers are recorded per stage.

    Raises :class:`JointFMConfigurationError` when the client retries or when
    the pool logger cannot emit warnings (failovers would go unseen), and
    :class:`JointFMCapacityError` when the deployment advertises no
    ``data_generation`` block. Errors raised while a stage runs are recorded
    in that stage instead, so the report still shows every stage.
    """
    if client.retry_config.max_attempts != 1:
        raise JointFMConfigurationError(
            "run_max_utilization_check requires a client built with "
            "JointFMRetryConfig(max_attempts=1); got "
            f"max_attempts={client.retry_config.max_attempts}"
        )
    pool_logger = logging.getLogger(POOL_LOGGER_NAME)
    if not pool_logger.isEnabledFor(logging.WARNING):
        raise JointFMConfigurationError(
            f"logger {POOL_LOGGER_NAME!r} must emit WARNING records so pool "
            "failovers are visible to the utilization check"
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
    history = _synthetic_history(plan.target_columns, envelope.n_input, seed=seed)
    query_times = list(range(envelope.n_input, envelope.n_input + envelope.n_output))
    batch_cap = health.max_sample_count

    single_request = _run_stage(
        SINGLE_REQUEST_STAGE,
        client,
        pool_logger=pool_logger,
        history=history,
        plan=plan,
        query_times=query_times,
        n_samples=batch_cap,
        batch_cap=batch_cap,
        seed=seed,
    )
    pool_saturation = None
    if len(instances.instances) > 1:
        pool_saturation = _run_stage(
            POOL_SATURATION_STAGE,
            client,
            pool_logger=pool_logger,
            history=history,
            plan=plan,
            query_times=query_times,
            n_samples=instances.max_sample_count,
            batch_cap=batch_cap,
            seed=seed,
        )

    return MaxUtilizationReport(
        envelope=envelope,
        decoding_strategy=health.decoding_strategy,
        topology_label=instances.topology_label,
        unavailable_instances=tuple(
            str(instance.deployment_id)
            for instance in instances.instances
            if instance.metadata is None
        ),
        read_timeout_seconds=client.timeout.read_seconds,
        health_seconds=health_seconds,
        single_request=single_request,
        pool_saturation=pool_saturation,
    )


class _FailoverRecorder(logging.Handler):
    """Collect pool warnings emitted while one stage runs."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted warning message."""
        self.messages.append(record.getMessage())


def _run_stage(
    name: str,
    client: JointFMClient,
    *,
    pool_logger: logging.Logger,
    history: Sequence[Mapping[str, float]],
    plan: ForecastPlan,
    query_times: Sequence[int],
    n_samples: int,
    batch_cap: int,
    seed: int,
) -> UtilizationStage:
    recorder = _FailoverRecorder()
    pool_logger.addHandler(recorder)
    error: str | None = None
    started = time.perf_counter()
    try:
        result = client.forecast_samples(
            history,
            columns=plan.columns,
            query_times=query_times,
            requested_columns=plan.requested_columns,
            n_samples=n_samples,
            seed=seed,
        )
        if len(result.samples) != n_samples:
            error = f"expected {n_samples} samples, got {len(result.samples)}"
    # A service or transport failure under maximum load is the verdict itself,
    # so it is recorded rather than propagated; other errors still raise.
    except JointFMError as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        wallclock_seconds = time.perf_counter() - started
        pool_logger.removeHandler(recorder)
    return UtilizationStage(
        name=name,
        n_samples=n_samples,
        request_count=math.ceil(n_samples / batch_cap),
        wallclock_seconds=wallclock_seconds,
        failovers=tuple(recorder.messages),
        error=error,
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
