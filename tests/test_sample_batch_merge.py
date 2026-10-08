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

"""Tests for merging split sample forecasts into one result."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import threading
import time
from typing import Any, cast
import weakref

import numpy as np
import pytest

import jointfm_client.client as client_module
from jointfm_client import (
    ColumnSpec,
    DataFrameSchema,
    ForecastResponse,
    JointFMClient,
    JointFMInstanceSettings,
    JointFMServiceError,
    JointFMSettings,
    SampleForecastResult,
)

_MODEL_VERSION = "jointfm-inference:0.3.0+ckpt.sdk-test"
_PRIMARY = (
    "https://app.datarobot.com/api/v2/deployments/primary-id/predictionsUnstructured"
)
_BACKUP = (
    "https://app.datarobot.com/api/v2/deployments/backup-id/predictionsUnstructured"
)
_QUERY_TIMES = [2, 3]
_SEED = 7


def _health_payload(*, max_sample_count: int) -> dict[str, object]:
    """Health payload advertising ``max_sample_count``."""
    return {
        "status": "ok",
        "schema_version": "v5",
        "image_version": "0.3.0",
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "sdk-test",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "head": "studentt",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": ["forecast"],
        "supported_condition_kinds": [],
        "supported_return_modes": ["mean", "samples", "quantiles", "log_prob"],
        "supported_time_index_modes": [
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ],
        "time_index_encoding": "legacy_discrete_grid",
        "max_sample_count": max_sample_count,
        "max_concurrent_requests": 1,
    }


def _batch_samples(seed: int, sample_count: int) -> list[list[list[float]]]:
    """Samples whose values name their batch seed, row, and horizon."""
    return [
        [[seed * 100.0 + row + horizon / 10.0] for horizon in range(len(_QUERY_TIMES))]
        for row in range(sample_count)
    ]


def _samples_response(
    seed: int, sample_count: int, *, image_version: str = "0.3.0"
) -> dict[str, object]:
    """One sample batch response answering ``seed`` and ``sample_count``."""
    return {
        "schema_version": "v5",
        "image_version": image_version,
        "model_version": _MODEL_VERSION,
        "checkpoint_version": "sdk-test",
        "head": "studentt",
        "query_mode": "forecast",
        "return_mode": "samples",
        "outputs": {
            "query_times": list(_QUERY_TIMES),
            "requested_columns": ["target"],
            "samples": _batch_samples(seed, sample_count),
        },
        "diagnostics": {
            "history_rows": 2,
            "horizon_count": len(_QUERY_TIMES),
            "seed": seed,
        },
        "errors": [],
    }


def _expected_samples(batch_sizes: list[int]) -> np.ndarray:
    """The batches concatenated in batch order, batch ``i`` drawn with seed ``7 + i``."""
    return np.concatenate(
        [
            np.asarray(_batch_samples(_SEED + index, size), dtype=np.float64)
            for index, size in enumerate(batch_sizes)
        ],
        axis=0,
    )


class _BatchTransport:
    """Answers health with a sample cap and every predict with seeded samples."""

    def __init__(self, *, mismatched_seed: int | None = None) -> None:
        """Init; the batch drawn with ``mismatched_seed`` reports another image."""
        self.mismatched_seed = mismatched_seed
        self.batches: list[tuple[int, int]] = []
        self.before_predict: dict[int, threading.Event] = {}
        self.observe_live_batches: Callable[[], list[int]] = list
        self.live_batches_during_predict: dict[int, list[int]] = {}

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        return _health_payload(max_sample_count=2)

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        if payload.get("request_type") == "health":
            return _health_payload(max_sample_count=2)
        seed = cast(int, payload["seed"])
        sample_count = cast(int, payload["n_samples"])
        gate = self.before_predict.get(seed)
        if gate is not None:
            assert gate.wait(timeout=10), f"batch with seed {seed} was never released"
        live_batches = self.observe_live_batches()
        # A gate opens while the merger is still copying the batch it waited
        # for, so give the caller a moment to let that batch go.
        deadline = time.monotonic() + 5
        while gate is not None and live_batches and time.monotonic() < deadline:
            time.sleep(0.01)
            live_batches = self.observe_live_batches()
        self.live_batches_during_predict[seed] = live_batches
        self.batches.append((seed, sample_count))
        image_version = "0.4.0" if seed == self.mismatched_seed else "0.3.0"
        return _samples_response(seed, sample_count, image_version=image_version)


def _single_endpoint_client(transport: _BatchTransport) -> JointFMClient:
    """Client bound to one local endpoint."""
    settings = JointFMSettings(
        datarobot_endpoint=None,
        datarobot_api_token=None,
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        deployment_selector="local_service",
        schema_version="v5",
        model_version=_MODEL_VERSION,
        local_base_url="http://127.0.0.1:8080",
    )
    return JointFMClient(settings=settings, transport=transport)


class _PeerTransport:
    """One peer's own transport, so the pool lets peers post concurrently."""

    def __init__(self, shared: _BatchTransport) -> None:
        """Init."""
        self.shared = shared

    def get_json(self, url: str) -> Mapping[str, Any]:
        """Get json."""
        return self.shared.get_json(url)

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post json."""
        return self.shared.post_json(url, payload)


class _PerPeerTransportClient(JointFMClient):
    """Client building each pool peer's transport as a view of one shared transport."""

    def __init__(self, *, settings: JointFMSettings, shared: _BatchTransport) -> None:
        """Init."""
        super().__init__(settings=settings)
        self._shared = shared

    def _new_pool_peer_transport(self) -> _PeerTransport:
        """Return a distinct transport for one peer."""
        return _PeerTransport(self._shared)


def _pool_client(transport: _BatchTransport) -> JointFMClient:
    """Client spreading sample batches over two pooled deployments.

    The pool serializes peers that share one transport, so each peer gets its
    own view of ``transport``.
    """
    settings = JointFMSettings(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="secret-token",
        health_url=_PRIMARY,
        predict_url=_PRIMARY,
        deployment_selector="deployment_ids",
        schema_version="v5",
        instances=(
            JointFMInstanceSettings(deployment_id="primary-id", predict_url=_PRIMARY),
            JointFMInstanceSettings(deployment_id="backup-id", predict_url=_BACKUP),
        ),
        model_version=_MODEL_VERSION,
        deployment_id="primary-id",
    )
    return _PerPeerTransportClient(settings=settings, shared=transport)


def _forecast_samples(client: JointFMClient, n_samples: int) -> SampleForecastResult:
    """Request ``n_samples`` seeded samples over two horizons of one target."""
    result = client.forecast_samples(
        [{"target": 10.0}, {"target": 11.0}],
        schema=DataFrameSchema(
            columns=(ColumnSpec(name="target", modality="numeric", role="target"),),
            time_index_mode="ordinal",
        ),
        query_times=list(_QUERY_TIMES),
        requested_columns=["target"],
        model_version=_MODEL_VERSION,
        n_samples=n_samples,
        seed=_SEED,
    )
    assert isinstance(result, SampleForecastResult)
    return result


class _AddRecorder:
    """Wraps the merger's ``add`` to record arrival order and batch lifetimes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init."""
        self.order: list[int] = []
        self.added: dict[int, threading.Event] = {}
        self.alive_at_add: list[list[int]] = []
        self._batch_refs: dict[int, weakref.ref[np.ndarray]] = {}
        original_add = client_module._SampleBatchMerger.add

        def _add(
            merger: Any, batch_index: int, batch_result: SampleForecastResult
        ) -> None:
            self.alive_at_add.append(self.live_batches())
            self._batch_refs[batch_index] = weakref.ref(batch_result.samples)
            self.order.append(batch_index)
            original_add(merger, batch_index, batch_result)
            self.added.setdefault(batch_index, threading.Event()).set()

        monkeypatch.setattr(client_module._SampleBatchMerger, "add", _add)

    def live_batches(self) -> list[int]:
        """Indexes of merged batches whose samples are still alive."""
        return [index for index, ref in self._batch_refs.items() if ref() is not None]


def test_sequential_batches_fill_the_merged_array_in_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped endpoint's batches merge into their concatenation, read-only."""
    recorder = _AddRecorder(monkeypatch)
    transport = _BatchTransport()
    transport.observe_live_batches = recorder.live_batches

    result = _forecast_samples(_single_endpoint_client(transport), n_samples=5)

    assert transport.batches == [(7, 2), (8, 2), (9, 1)]
    np.testing.assert_array_equal(result.samples, _expected_samples([2, 2, 1]))
    assert result.samples.dtype == np.float64
    assert not result.samples.flags.writeable
    assert result.diagnostics.seed == _SEED
    assert result.diagnostics.horizon_count == len(_QUERY_TIMES)
    assert result.query_times == tuple(_QUERY_TIMES)
    assert result.errors == ()
    # Each batch's samples are gone before the next batch is fetched.
    assert transport.live_batches_during_predict == {7: [], 8: [], 9: []}
    assert recorder.alive_at_add == [[], [], []]


def test_pool_batches_merge_in_batch_order_whatever_order_they_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch 0 finishing after batch 1 still lands in the first rows."""
    recorder = _AddRecorder(monkeypatch)
    transport = _BatchTransport()
    # Hold batch 0 (seed 7) on its peer until batch 1 has been merged.
    transport.before_predict[_SEED] = recorder.added.setdefault(1, threading.Event())
    transport.observe_live_batches = recorder.live_batches

    result = _forecast_samples(_pool_client(transport), n_samples=5)

    assert recorder.order[0] == 1
    # Batch 1 is merged and freed while batch 0 is still being fetched.
    assert transport.live_batches_during_predict[_SEED] == []
    assert sorted(recorder.order) == [0, 1, 2]
    np.testing.assert_array_equal(result.samples, _expected_samples([2, 2, 1]))
    assert not result.samples.flags.writeable
    assert result.diagnostics.seed == _SEED
    assert recorder.alive_at_add == [[], [], []]


@pytest.mark.parametrize("make_client", [_single_endpoint_client, _pool_client])
def test_a_batch_disagreeing_with_the_first_is_refused(make_client: Any) -> None:
    """Every batch must answer with the first batch's metadata."""
    transport = _BatchTransport(mismatched_seed=_SEED + 2)

    with pytest.raises(
        JointFMServiceError,
        match=(
            "JointFM forecast response violated the service contract: "
            "sample batch image_version mismatch"
        ),
    ):
        _forecast_samples(make_client(transport), n_samples=5)


def _batch_result(seed: int, sample_count: int) -> SampleForecastResult:
    """Parse one sample batch response."""
    result = ForecastResponse.from_payload(_samples_response(seed, sample_count))
    assert isinstance(result, SampleForecastResult)
    return result


def test_a_missing_batch_is_refused_as_the_wrong_sample_count() -> None:
    """The merged rows must add up to the requested sample count."""
    merger = client_module._SampleBatchMerger({"n_samples": 4}, [2, 2])
    merger.add(0, _batch_result(_SEED, 2))

    with pytest.raises(
        JointFMServiceError,
        match="sample batching produced the wrong sample count: expected 4, got 2",
    ):
        merger.result()


def test_a_batch_that_does_not_fit_its_rows_is_refused() -> None:
    """A short batch would otherwise broadcast over its rows."""
    merger = client_module._SampleBatchMerger({"n_samples": 4}, [2, 2])
    merger.add(0, _batch_result(_SEED, 2))

    with pytest.raises(
        JointFMServiceError,
        match=r"sample batch 1 shape mismatch: expected \(2, 2, 1\), got \(1, 2, 1\)",
    ):
        merger.add(1, _batch_result(_SEED + 1, 1))


def test_merging_no_batches_is_refused() -> None:
    """A merge with nothing to merge is a contract violation, not an empty result."""
    merger = client_module._SampleBatchMerger({"n_samples": 2}, [2])

    with pytest.raises(
        JointFMServiceError, match="sample batching produced no responses"
    ):
        merger.result()
