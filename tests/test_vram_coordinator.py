"""
Tests for VRAM Coordinator.

The suite covers the behaviours the module documents: the cumulative
budget, the rules for trusting the lock-free measurement, the model layer
contract, the instance lifecycle, eviction and the health machine, and the
concurrency guarantees of the watchdog.

The sysfs tree is faked in every test that depends on it, so results do not
depend on the machine running the suite. Hardware facts used below are from
an RX 580 (8 GB VRAM, 256 MB BAR window).
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import vram_coordinator
from vram_coordinator import (
    CONTENDED_READ_MIN_INTERVAL_S,
    HEALTH_CHECK_FAILURE_THRESHOLD,
    AcquisitionResult,
    GPUHealth,
    ModelSpec,
    VRAMCoordinator,
    reset_coordinator,
    shutdown_coordinator,
)

# --- Helpers ---

MB = 1024 * 1024


def _make_spec(
    model_id: str = "/tmp/test_model.gguf",
    estimated_mb: int = 1024,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    total_layers: int | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        estimated_mb=estimated_mb,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        total_layers=total_layers,
    )


class FakeModel:
    """Stand-in for a llama.cpp model, recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --- Test fixtures ---


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset coordinator singleton and reader bookkeeping between tests."""
    reset_coordinator()
    with vram_coordinator._reads_lock:
        vram_coordinator._reads.clear()
        vram_coordinator._timeout_reported.clear()
    yield
    shutdown_coordinator()


@pytest.fixture
def fake_sysfs(tmp_path, monkeypatch):
    """Point the coordinator at a fake amdgpu sysfs device directory.

    Returns a builder so each test can describe the device it wants: one whose
    visible window covers VRAM, or an RX 580 style device whose window is a
    fraction of it. A value of ``None`` means the attribute is not created,
    which is how a missing attribute is simulated.
    """

    def _build(
        *,
        vram_total_mb: int | None = 8192,
        vis_total_mb: int | None = 8192,
        vis_used_mb: int | None = 0,
        vram_used_mb: int | None = None,
    ) -> Path:
        device = tmp_path / "device"
        device.mkdir(exist_ok=True)
        attributes = {
            vram_coordinator._ATTR_VRAM_TOTAL: vram_total_mb,
            vram_coordinator._ATTR_VIS_VRAM_TOTAL: vis_total_mb,
            vram_coordinator._ATTR_VIS_VRAM_USED: vis_used_mb,
            vram_coordinator._ATTR_VRAM_USED: vram_used_mb,
        }
        for name, value_mb in attributes.items():
            path = device / name
            if value_mb is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(str(value_mb * MB), encoding="utf-8")

        monkeypatch.setattr(vram_coordinator, "_DRM_DEVICE_GLOB", str(device))
        return device

    return _build


# --- Tests: VRAM detection and measurement ---


def test_detects_total_vram_from_sysfs(fake_sysfs):
    """Total VRAM comes from the static lock-free sysfs file."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192)
    coordinator = VRAMCoordinator()
    assert coordinator._total_vram_mb == 8192
    assert coordinator._vram_detected_from_sysfs is True


def test_falls_back_to_8192_when_sysfs_missing(fake_sysfs):
    """A host without amdgpu sysfs keeps working with the documented fallback."""
    fake_sysfs(vram_total_mb=None, vis_total_mb=None, vis_used_mb=None)
    coordinator = VRAMCoordinator()
    assert coordinator._total_vram_mb == 8192
    assert coordinator._vram_detected_from_sysfs is False
    assert coordinator._visible_window_trustworthy is False


def test_free_vram_is_exact_when_window_covers_card(fake_sysfs):
    """With a full visible window the lock-free read backs the budget."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=2048)
    coordinator = VRAMCoordinator()
    # 8192 - 2048 used - 1024 system reserve - 256 headroom
    assert coordinator._get_free_vram_mb() == 4864
    assert coordinator._measurement_source() == "lockfree_visible_window"


def test_free_vram_is_none_when_visible_window_is_too_small(fake_sysfs):
    """An RX 580 style device cannot back a budget from the visible window.

    mem_info_vis_vram_used is bounded by the BAR aperture (256 MB here) while
    the device holds 8 GB and 5 GB are in use. Trusting it would report about
    6.7 GB free and therefore over-commit the device, so the coordinator
    reports that it cannot measure instead.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=4988)
    coordinator = VRAMCoordinator()
    assert coordinator._visible_window_trustworthy is False
    assert coordinator._measurement_source() == "unavailable"
    assert coordinator._get_free_vram_mb() is None


def test_contended_opt_in_reads_real_usage(fake_sysfs):
    """allow_contended_read=True uses mem_info_vram_used and reports real numbers."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=4988)
    coordinator = VRAMCoordinator(allow_contended_read=True)
    # 8192 - 4988 used - 1024 reserve - 256 headroom
    assert coordinator._get_free_vram_mb() == 1924
    assert coordinator._measurement_source() == "contended_mem_info_vram_used"


def test_contended_read_is_rate_limited(fake_sysfs, monkeypatch):
    """The contended path is read at most once per interval, then cached."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=1024)
    reads: list[str] = []
    real_read = vram_coordinator._read_sysfs_int

    def _spy(path_str: str, timeout_s: float) -> int | None:
        reads.append(path_str)
        return real_read(path_str, timeout_s)

    monkeypatch.setattr(vram_coordinator, "_read_sysfs_int", _spy)
    coordinator = VRAMCoordinator(allow_contended_read=True)
    assert coordinator._get_free_vram_mb() == 8192 - 1024 - 1024 - 256
    contended_reads = [p for p in reads if p.endswith("mem_info_vram_used")]
    assert len(contended_reads) == 1
    # A second call inside the interval must reuse the cached value
    assert coordinator._get_free_vram_mb() == 8192 - 1024 - 1024 - 256
    assert len([p for p in reads if p.endswith("mem_info_vram_used")]) == 1
    assert CONTENDED_READ_MIN_INTERVAL_S > 0


def test_contended_path_is_never_read_by_default(fake_sysfs, monkeypatch):
    """The lock-contended attribute is not read unless the operator enables it.

    This is the property the module promises: in the default configuration,
    no hardware-dependent path reads mem_info_vram_used.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=4988)
    reads: list[str] = []
    real_read = vram_coordinator._read_sysfs_int

    def _spy(path_str: str, timeout_s: float) -> int | None:
        reads.append(path_str)
        return real_read(path_str, timeout_s)

    monkeypatch.setattr(vram_coordinator, "_read_sysfs_int", _spy)
    coordinator = VRAMCoordinator()
    coordinator.acquire(_make_spec(estimated_mb=512))
    coordinator._run_watchdog_iteration()
    assert not [p for p in reads if p.endswith("mem_info_vram_used")]


def test_acquire_denies_when_window_too_small(fake_sysfs):
    """End to end: an untrustworthy window means CPU fallback, never a guess."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=4988)
    coordinator = VRAMCoordinator()
    result = coordinator.acquire(_make_spec(estimated_mb=512))
    assert result.granted is False
    assert result.fallback_to_cpu is True
    assert "unavailable" in result.reason
    assert "visible_vram_window_too_small" in result.reason


def _count_reader_starts(monkeypatch) -> list[str]:
    """Record the name of every thread started while the test runs."""
    started: list[str] = []
    real_start = threading.Thread.start

    def counting_start(self: threading.Thread) -> None:
        started.append(self.name)
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", counting_start)
    return started


def _reader_starts(started: list[str]) -> int:
    """How many sysfs readers were started."""
    return started.count("VRAMSysfsRead")


def _release_blocked_reader(path: Path) -> None:
    """Unblock a reader parked on a FIFO, so the test leaves no thread behind."""
    fd = os.open(path, os.O_WRONLY)
    os.write(fd, b"0")
    os.close(fd)


def test_sysfs_read_times_out_instead_of_blocking(tmp_path):
    """A wedged driver must not trap the watchdog on an unbounded read."""
    blocked = tmp_path / "mem_info_blocked"
    os.mkfifo(blocked)
    started = time.monotonic()
    assert vram_coordinator._read_sysfs_int(str(blocked), 0.2) is None
    assert time.monotonic() - started < 1.0
    _release_blocked_reader(blocked)


def test_repeated_timeouts_do_not_pile_up_readers(tmp_path, monkeypatch):
    """A blocked attribute starts one reader, not one per call."""
    blocked = tmp_path / "mem_info_blocked"
    os.mkfifo(blocked)
    started = _count_reader_starts(monkeypatch)

    for _ in range(5):
        assert vram_coordinator._read_sysfs_int(str(blocked), 0.05) is None

    assert _reader_starts(started) == 1
    _release_blocked_reader(blocked)


def test_concurrent_reads_of_a_healthy_attribute_share_the_result(tmp_path):
    """Sharing a read must not turn into a spurious "cannot measure"."""
    target = tmp_path / "mem_info_vram_total"
    target.write_text(str(8192 * MB), encoding="utf-8")
    results: list[int | None] = []
    results_lock = threading.Lock()

    def read() -> None:
        value = vram_coordinator._read_sysfs_int(str(target), 1.0)
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=read, daemon=True) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(results) == 8
    assert set(results) == {8192 * MB}


def test_concurrent_reads_of_a_blocked_attribute_share_one_reader(tmp_path, monkeypatch):
    """A burst on a blocked path starts one reader, not one per caller."""
    blocked = tmp_path / "mem_info_blocked"
    os.mkfifo(blocked)
    started = _count_reader_starts(monkeypatch)
    results: list[int | None] = []
    results_lock = threading.Lock()

    def read() -> None:
        value = vram_coordinator._read_sysfs_int(str(blocked), 0.2)
        with results_lock:
            results.append(value)

    threads = [
        threading.Thread(target=read, daemon=True, name=f"caller-{index}") for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert _reader_starts(started) == 1
    assert len(results) == 8
    assert all(value is None for value in results)
    _release_blocked_reader(blocked)


def test_measurement_source_reports_unavailable_when_read_fails(fake_sysfs):
    """The reported source describes the read that actually happened."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233)  # no vram_used
    coordinator = VRAMCoordinator(allow_contended_read=True)
    assert coordinator._get_free_vram_mb() is None
    assert coordinator._measurement_source() == "unavailable"

    result = coordinator.acquire(_make_spec(estimated_mb=64))
    assert result.granted is False
    assert "could not be read" in result.reason
    assert "allow_contended_read=True" not in result.reason


# --- Tests: VRAMCoordinator acquire/release ---


def test_acquire_grants_when_vram_available(fake_sysfs):
    """If free_mb >= estimated_mb, the reservation is granted."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=2048)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    result = coordinator.acquire(spec)
    assert result.granted is True
    assert result.reservation_id is not None
    assert result.gpu_layers_actual == -1
    coordinator.release(result.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 0


def test_acquire_denies_when_vram_exhausted():
    """If it requested "everything" (n_gpu_layers=-1) and does not fit, deny with fallback."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=64):
        spec = _make_spec(estimated_mb=4096)  # way too large for 64MB
        result = coordinator.acquire(spec)
        assert result.granted is False
        assert result.fallback_to_cpu is True
        assert result.reason.startswith("model_too_large_for_full_gpu")


def test_acquire_denies_when_sysfs_unavailable():
    """If VRAM cannot be measured (sysfs unavailable), deny with fallback."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=None):
        result = coordinator.acquire(_make_spec())
        assert result.granted is False
        assert result.fallback_to_cpu is True
        assert "unavailable" in result.reason.lower()


def test_budget_is_cumulative(fake_sysfs):
    """Granted reservations are subtracted, so loads cannot over-commit.

    Four 800 MB requests against a 1280 MB budget yield a single grant: each
    caller sees the budget the previous one left.
    """
    fake_sysfs(vram_total_mb=2560, vis_total_mb=2560, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    # free = 2560 - 0 - 1024 - 256 = 1280 MB
    assert coordinator._get_free_vram_mb() == 1280

    granted = 0
    for index in range(4):
        result = coordinator.acquire(
            _make_spec(model_id=f"/tmp/model-{index}.gguf", estimated_mb=800, n_ctx=2048)
        )
        granted += int(result.granted)

    assert granted == 1, "only the first reservation fits in a 1280MB budget"
    metrics = coordinator.get_metrics()
    assert metrics["vram_reserved_mb"] == 801  # 800 MB + 1 MB of KV cache
    assert metrics["allocations"] == 1


def test_reservations_never_exceed_the_budget(fake_sysfs):
    """Invariant: reserved + used + reserve + headroom <= total."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=2048)
    coordinator = VRAMCoordinator()
    for index in range(20):
        coordinator.acquire(
            _make_spec(model_id=f"/tmp/small-{index}.gguf", estimated_mb=256, n_ctx=2048)
        )
    metrics = coordinator.get_metrics()
    committed = metrics["vram_reserved_mb"] + 2048 + 1024 + 256
    assert committed <= coordinator._total_vram_mb


def test_release_of_unknown_handle_is_ignored():
    """An unknown handle must not decrement somebody else's live allocation."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        result = coordinator.acquire(_make_spec(estimated_mb=512))
        assert result.granted
        ref_before = coordinator._allocations["/tmp/test_model.gguf"].ref_count
        coordinator.release("/tmp/does-not-exist.gguf::12345")
        assert coordinator._allocations["/tmp/test_model.gguf"].ref_count == ref_before


def test_release_twice_does_not_go_negative():
    """A stale handle is reported and ignored instead of corrupting the count."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        result = coordinator.acquire(_make_spec(estimated_mb=512))
        coordinator.release(result.reservation_id)
        assert coordinator._allocations["/tmp/test_model.gguf"].ref_count == 0
        coordinator.release(result.reservation_id)  # stale handle
        assert coordinator._allocations["/tmp/test_model.gguf"].ref_count == 0
        assert coordinator.get_metrics()["releases"] == 1


def test_acquire_reuses_existing_allocation():
    """A second call with the same model_id reuses the instance (refcount)."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        spec = _make_spec(estimated_mb=512)
        first = coordinator.acquire(spec)
        assert first.granted is True
        assert first.reservation_id is not None
        second = coordinator.acquire(spec)
        assert second.granted is True
        assert second.reservation_id is not None
        assert "shared" in second.reservation_id
        alloc = coordinator._allocations[spec.model_id]
        assert alloc.ref_count == 2
        coordinator.release(first.reservation_id)
        coordinator.release(second.reservation_id)
        assert alloc.ref_count == 0


def test_shared_grant_is_flagged_and_gets_its_own_handle(fake_sysfs):
    """A grant that shares an already loaded model is marked as such.

    The flat API cannot hand out the existing instance, so it flags the grant:
    the caller must reuse the loaded model instead of loading another copy.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=1024)

    first = coordinator.acquire(spec)
    assert first.granted is True
    assert first.already_loaded is False, "the caller loads this model itself"

    second = coordinator.acquire(spec)
    assert second.granted is True
    assert second.already_loaded is True, "another holder already has this model"
    assert second.reservation_id != first.reservation_id
    assert coordinator._allocations[spec.model_id].ref_count == 2
    assert coordinator.get_metrics()["vram_reserved_mb"] == 1025

    coordinator.release(first.reservation_id)
    coordinator.release(second.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 0


def test_double_release_of_a_shared_handle_spares_other_holders(fake_sysfs):
    """A repeated release is ignored, so it cannot drop another holder's reference."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)

    first = coordinator.acquire(spec)
    second = coordinator.acquire(spec)
    assert coordinator._allocations[spec.model_id].ref_count == 2

    coordinator.release(second.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 1

    coordinator.release(second.reservation_id)  # already released
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert coordinator.get_metrics()["releases"] == 1

    coordinator.release(first.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 0


# --- Tests: GPU layer semantics ---


def test_explicit_layers_are_honored_as_is(fake_sysfs):
    """An explicit n_gpu_layers value is handed to the loader unchanged."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512, n_gpu_layers=10, total_layers=32)
    result = coordinator.acquire(spec)
    assert result.granted is True
    assert result.gpu_layers_actual == 10


def test_all_layers_request_is_not_reduced_implicitly(fake_sysfs):
    """A request for all layers either fits or is denied.

    The request is not converted into a partial offload: a caller that wants
    partial GPU passes an explicit layer count.
    """
    fake_sysfs(vram_total_mb=2560, vis_total_mb=2560, vis_used_mb=0)
    coordinator = VRAMCoordinator()  # free = 1280 MB
    result = coordinator.acquire(_make_spec(estimated_mb=4096, n_gpu_layers=-1))
    assert result.granted is False
    assert result.gpu_layers_actual == 0
    assert "model_too_large_for_full_gpu" in result.reason


def test_estimate_matches_layer_fraction():
    """An explicit layer count is a fraction of the model, not a percentage."""
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=5000, n_ctx=4096, n_gpu_layers=30, total_layers=32)
    estimated = coordinator._estimate_vram_mb(spec, 30)
    # 30/32 of the model = 4687 MB, plus 2 MB of KV cache
    assert estimated == 4687 + 2


def test_estimate_without_total_layers_budgets_whole_model():
    """Unknown layer count must over-reserve, never under-reserve."""
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=5000, n_ctx=2048, n_gpu_layers=4)
    assert coordinator._estimate_vram_mb(spec, 4) == 5001


@pytest.mark.parametrize("layers", [0, 1, 10, 29, 33, 99, 100, 101, 999])
def test_estimate_never_exceeds_model_size(layers):
    """The estimate is bounded by the model size for every layer count.

    A layer count is read as layers, never as a percentage, and the estimate
    for a partial offload never exceeds the whole model.
    """
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=4096, n_ctx=4096, n_gpu_layers=layers, total_layers=33)
    estimated = coordinator._estimate_vram_mb(spec, layers)
    expected_layers = min(layers, 33)
    assert estimated == int(4096 * expected_layers / 33) + 2


def test_estimate_is_monotonic_in_layers():
    """More layers offloaded never costs less VRAM."""
    coordinator = VRAMCoordinator()
    estimates = [
        coordinator._estimate_vram_mb(
            _make_spec(estimated_mb=4096, n_ctx=2048, total_layers=40), layers
        )
        for layers in range(45)
    ]
    assert estimates == sorted(estimates)


def test_high_layer_count_offloads_everything(fake_sysfs):
    """n_gpu_layers=99 on a 33 layer model means "all", not "99% of a 4 GB model"."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=4000, n_ctx=2048, n_gpu_layers=99, total_layers=33)
    result = coordinator.acquire(spec)
    assert result.granted is True
    assert coordinator._allocations[spec.model_id].reserved_mb == 4001


def test_estimate_vram_includes_context_overhead():
    """_estimate_vram_mb adds context overhead (KV cache)."""
    coordinator = VRAMCoordinator()
    vram = coordinator._estimate_vram_mb(
        _make_spec(estimated_mb=1000, n_ctx=4096), gpu_layers_actual=-1
    )
    assert vram >= 1000  # base
    assert vram >= 1002  # +2MB ctx overhead


# --- Tests: Embedding service (shared) ---


def test_embedding_service_shares_instance():
    """Multiple get_or_create calls for the same model_id return the SAME instance."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    counter = {"calls": 0}

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> FakeModel:
        counter["calls"] += 1
        return FakeModel()

    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        spec = _make_spec(estimated_mb=512)
        instance1, result1 = service.get_or_create(spec, factory)
        instance2, result2 = service.get_or_create(spec, factory)
        assert counter["calls"] == 1
        assert instance1 is instance2
        assert result1.granted and result2.granted
        service.release(spec.model_id)


def test_embedding_service_factory_failure_returns_none():
    """If the factory returns None, get_or_create returns (None, error)."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> None:
        return None

    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        spec = _make_spec(estimated_mb=512)
        instance, result = service.get_or_create(spec, factory)
        assert instance is None
        assert result.granted is False
        assert result.reason == "factory_returned_none"
        assert result.fallback_to_cpu is True


def test_factory_failure_leaves_no_phantom_allocation():
    """A failed load must not keep consuming budget."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        spec = _make_spec(estimated_mb=2048)
        service.get_or_create(spec, lambda s, n: None)
        metrics = coordinator.get_metrics()
        assert metrics["allocations"] == 0
        assert metrics["vram_reserved_mb"] == 0
        assert coordinator._reservations == {}


def test_release_then_get_or_create_reloads(fake_sysfs):
    """Releasing the last reference and loading again starts from scratch.

    The released instance is closed, so the next call loads a new model
    instead of returning one the budget no longer accounts for.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    created: list[FakeModel] = []

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> FakeModel:
        model = FakeModel()
        created.append(model)
        return model

    spec = _make_spec(estimated_mb=512)
    first, _ = service.get_or_create(spec, factory)
    service.release(spec.model_id)
    assert spec.model_id not in coordinator._allocations
    assert first.closed is True, "the released model must be closed"
    assert spec.model_id not in service._instances

    second, result = service.get_or_create(spec, factory)  # must not raise
    assert result.granted is True
    assert second is not first
    assert len(created) == 2
    service.release(spec.model_id)


def test_get_or_create_rejects_instance_without_allocation():
    """A cached instance whose allocation vanished is stale, not reusable."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    stale = FakeModel()
    spec = _make_spec(estimated_mb=512)
    service._instances[spec.model_id] = stale  # simulate an eviction desync
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        instance, result = service.get_or_create(spec, lambda s, n: FakeModel())
        assert result.granted is True
        assert instance is not stale
        assert stale.closed is True
        service.release(spec.model_id)


def test_stale_instance_is_replaced_on_reuse(fake_sysfs):
    """A cached instance whose allocation vanished is replaced, not handed out.

    The instance cache and the allocation table must agree: a caller never
    receives a model that the evictor is closing.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    created: list[FakeModel] = []

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> FakeModel:
        model = FakeModel()
        created.append(model)
        return model

    spec = _make_spec(estimated_mb=512)
    first, _ = service.get_or_create(spec, factory)

    # Simulate the eviction winning the race: instance cached, allocation gone.
    coordinator._allocations.pop(spec.model_id, None)
    second, result = service.get_or_create(spec, factory)

    assert result.granted is True
    assert second is not first, "the stale instance must not be reused"
    assert first.closed is True
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert coordinator.get_metrics()["vram_reserved_mb"] == 513
    service.release(spec.model_id)


def test_service_refuses_to_duplicate_a_model_loaded_outside_it(fake_sysfs):
    """A model held through the flat API is not loaded a second time.

    The service cannot share an instance it does not own, so it reports
    ``model_loaded_outside_the_service`` and leaves the holder's reservation
    intact instead of loading another copy.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    spec = _make_spec(estimated_mb=512)

    held = coordinator.acquire(spec)
    assert held.granted is True
    assert held.already_loaded is False

    calls = {"count": 0}

    def factory(s: ModelSpec, n: int) -> FakeModel:
        calls["count"] += 1
        return FakeModel()

    instance, result = service.get_or_create(spec, factory)

    assert instance is None
    assert result.granted is False
    assert result.reason == "model_loaded_outside_the_service"
    assert calls["count"] == 0, "no second copy may be loaded"
    assert coordinator._allocations[spec.model_id].ref_count == 1, "the holder keeps its reference"
    assert coordinator.get_metrics()["vram_reserved_mb"] == 513

    coordinator.release(held.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 0


def test_eviction_closes_the_instance(fake_sysfs):
    """Evicting a model must close it, not just forget it."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)

    def factory(s: ModelSpec, n: int) -> FakeModel:
        model = FakeModel()
        model.id = s.model_id
        return model

    with patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0):
        service = coordinator.get_embedding_service()
        model, result = service.get_or_create(spec, factory)
        assert result.granted
        coordinator.release(result.reservation_id)  # ref_count -> 0
        freed = coordinator._evict_idle(1)
        assert freed > 0
        assert model.closed is True
        assert spec.model_id not in coordinator._allocations
        assert spec.model_id not in service._instances
        assert coordinator._reservations == {}, "an evicted model must not keep handles"
        assert coordinator._evictions == 1


def test_eviction_from_acquire_closes_without_the_coordinator_lock(fake_sysfs):
    """Eviction closes the models it drops with the coordinator lock released.

    Closing takes the service lock and the service takes the coordinator lock,
    so the two must never nest in that order.
    """
    fake_sysfs(vram_total_mb=4096, vis_total_mb=4096, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    lock_held_while_closing: list[bool] = []
    real_discard = service._discard

    def spy_discard(model_id: str) -> None:
        lock_held_while_closing.append(coordinator._lock._is_owned())
        real_discard(model_id)

    with (
        patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0),
        patch.object(service, "_discard", side_effect=spy_discard),
    ):
        idle_spec = _make_spec(model_id="/tmp/idle.gguf", estimated_mb=512)
        idle_model, idle_result = service.get_or_create(idle_spec, lambda s, n: FakeModel())
        coordinator.release(idle_result.reservation_id)

        # free is 2816 MB, so a request of exactly that size passes the
        # all-layers check and only exceeds the budget by the KV cache
        # overhead: that is the path that attempts eviction.
        oversized = _make_spec(model_id="/tmp/oversized.gguf", estimated_mb=2816, n_ctx=2048)
        coordinator.acquire(oversized)

    assert lock_held_while_closing, "the eviction path did not run"
    assert not any(lock_held_while_closing), "the coordinator lock was held while closing"
    assert idle_model.closed is True


def test_eviction_from_the_watchdog_closes_without_the_coordinator_lock(fake_sysfs):
    """The watchdog closes evicted models with the coordinator lock released."""
    device = fake_sysfs(vram_total_mb=4096, vis_total_mb=4096, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    lock_held_while_closing: list[bool] = []
    real_discard = service._discard

    def spy_discard(model_id: str) -> None:
        lock_held_while_closing.append(coordinator._lock._is_owned())
        real_discard(model_id)

    with (
        patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0),
        patch.object(service, "_discard", side_effect=spy_discard),
    ):
        idle_spec = _make_spec(model_id="/tmp/idle.gguf", estimated_mb=512, n_ctx=2048)
        idle_model, idle_result = service.get_or_create(idle_spec, lambda s, n: FakeModel())
        assert idle_result.granted is True
        coordinator.release(idle_result.reservation_id)

        # The device fills up, so the next tick is under pressure and evicts.
        used = device / vram_coordinator._ATTR_VIS_VRAM_USED
        used.write_text(str(3800 * MB), encoding="utf-8")
        coordinator._run_watchdog_iteration()

    assert lock_held_while_closing, "the eviction path did not run"
    assert not any(lock_held_while_closing), "the coordinator lock was held while closing"
    assert idle_model.closed is True


def test_shared_grant_is_registered_without_the_caller_holding_the_lock(fake_sysfs):
    """The registration takes the coordinator lock itself.

    The existence check, the reference increment and the new handle are atomic
    with respect to an eviction, whichever side calls it.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    assert coordinator.acquire(spec).granted

    grants: list[tuple[str, int] | None] = []

    def register() -> None:
        assert coordinator._lock._is_owned() is False, "precondition: no lock held"
        grants.append(coordinator._register_shared_grant(spec.model_id))

    worker = threading.Thread(target=register, daemon=True)
    worker.start()
    worker.join(timeout=5.0)

    assert grants and grants[0] is not None
    handle, gpu_layers_actual = grants[0]
    assert gpu_layers_actual == -1
    assert handle in coordinator._reservations
    assert coordinator._allocations[spec.model_id].ref_count == 2
    coordinator.release(handle)
    assert coordinator._allocations[spec.model_id].ref_count == 1


def test_service_load_under_pressure_does_not_self_deadlock(fake_sysfs):
    """A service load that triggers eviction must not block on its own lock.

    ``get_or_create`` holds the service lock across the acquisition, and the
    acquisition can evict and close an instance through that same lock.
    """
    fake_sysfs(vram_total_mb=4096, vis_total_mb=4096, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()

    with patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0):
        # An evictable allocation that the service load will have to reclaim.
        idle_spec = _make_spec(model_id="/tmp/idle.gguf", estimated_mb=512, n_ctx=2048)
        idle_model, idle_result = service.get_or_create(idle_spec, lambda s, n: FakeModel())
        assert idle_result.granted is True
        coordinator.release(idle_result.reservation_id)

        # free is 2816 MB and the idle model still holds 513 MB of it, so this
        # request only fits after the eviction above.
        loaded: dict[str, object] = {}

        def load() -> None:
            spec = _make_spec(model_id="/tmp/needy.gguf", estimated_mb=2500, n_ctx=2048)
            loaded["value"] = service.get_or_create(spec, lambda s, n: FakeModel())

        worker = threading.Thread(target=load, daemon=True)
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "the service load blocked on its own lock"

    _instance, result = loaded["value"]
    assert result.granted is True
    assert idle_model.closed is True


def test_deferred_discard_spares_an_instance_that_was_loaded_again(fake_sysfs):
    """Eviction must not close a model that has been acquired again.

    The allocation is dropped under the coordinator lock and the instance is
    closed afterwards; a model loaded in that window belongs to the new
    reservation and is left alone.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    spec = _make_spec(estimated_mb=512, n_ctx=2048)

    with patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0):
        first, first_result = service.get_or_create(spec, lambda s, n: FakeModel())
        coordinator.release(first_result.reservation_id)

        # The eviction collects the model but has not closed it yet.
        freed_mb, discarded = coordinator._collect_evictions(1)
        assert freed_mb > 0 and discarded == [spec.model_id]

        # A caller loads the same model again inside that window.
        second, second_result = service.get_or_create(spec, lambda s, n: FakeModel())
        assert second_result.granted is True

        # The deferred close must leave the newly loaded model alone.
        service._discard(spec.model_id)

    assert second.closed is False
    assert service._instances[spec.model_id] is second
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert first.closed is True
    service.release(spec.model_id)


def test_stale_handle_does_not_decrement_a_new_holder(fake_sysfs):
    """A handle from an earlier allocation cannot decrement a later holder.

    Every reference has its own handle, so a handle that was already released
    is ignored instead of decrementing a reference it does not own.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    spec = _make_spec(estimated_mb=512, n_ctx=2048)

    first, first_result = service.get_or_create(spec, lambda s, n: FakeModel())
    _second, second_result = service.get_or_create(spec, lambda s, n: FakeModel())
    assert coordinator._allocations[spec.model_id].ref_count == 2
    first_handle = first_result.reservation_id

    service.release(spec.model_id)  # one reference goes away
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert first_handle not in coordinator._reservations, "its handle must be revoked"

    coordinator.release(second_result.reservation_id)
    assert coordinator._allocations[spec.model_id].ref_count == 0
    assert coordinator._reservations == {}

    # A later holder must start with a clean reference count.
    third, third_result = service.get_or_create(spec, lambda s, n: FakeModel())
    assert third_result.granted is True
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert coordinator.release(first_handle) is None  # stale handle is ignored
    assert coordinator._allocations[spec.model_id].ref_count == 1
    service.release(spec.model_id)


def test_factory_that_raises_returns_the_reservation(fake_sysfs):
    """A loader that raises must not leave a reservation behind.

    Any failure to produce a model returns its reservation and handle, so the
    budget is not held and the model can be loaded again.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    spec = _make_spec(estimated_mb=512, n_ctx=2048)

    def exploding_factory(s: ModelSpec, n: int) -> FakeModel:
        raise RuntimeError("the loader failed")

    with pytest.raises(RuntimeError):
        service.get_or_create(spec, exploding_factory)

    metrics = coordinator.get_metrics()
    assert metrics["allocations"] == 0
    assert metrics["vram_reserved_mb"] == 0
    assert coordinator._reservations == {}

    # The model can be loaded again afterwards.
    model, result = service.get_or_create(spec, lambda s, n: FakeModel())
    assert result.granted is True
    assert model is not None
    service.release(spec.model_id)


def test_shared_grant_waits_for_the_coordinator_lock(fake_sysfs):
    """The registration takes the coordinator lock itself."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512, n_ctx=2048)
    assert coordinator.acquire(spec).granted

    holding = threading.Event()
    release = threading.Event()
    registered: list[object] = []

    def hold_the_lock() -> None:
        with coordinator._lock:
            holding.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    assert holding.wait(timeout=5.0)

    def register() -> None:
        registered.append(coordinator._register_shared_grant(spec.model_id))

    worker = threading.Thread(target=register, daemon=True)
    worker.start()
    worker.join(timeout=0.3)
    assert worker.is_alive(), "the registration must wait for the coordinator lock"

    release.set()
    worker.join(timeout=5.0)
    holder.join(timeout=5.0)

    assert registered and registered[0] is not None


def test_shutdown_releases_the_accounting(fake_sysfs):
    """After shutdown the coordinator holds nothing, so a new acquire is fresh."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    assert coordinator.acquire(spec).granted

    coordinator.shutdown()

    metrics = coordinator.get_metrics()
    assert metrics["allocations"] == 0
    assert metrics["vram_reserved_mb"] == 0
    fresh = coordinator.acquire(spec)
    assert fresh.granted is True
    assert fresh.already_loaded is False, "shutdown must not leave a phantom holder"


def test_abandon_keeps_another_holders_handle(fake_sysfs):
    """A failed load returns only its own reservation and handle."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    first = coordinator.acquire(spec)
    second = coordinator.acquire(spec)
    assert second.already_loaded is True

    coordinator._abandon_allocation(spec.model_id, first.reservation_id)

    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert first.reservation_id not in coordinator._reservations
    assert second.reservation_id in coordinator._reservations, "the survivor keeps a valid handle"
    coordinator.release(second.reservation_id)
    assert coordinator._reservations == {}


def test_thread_start_failure_degrades_to_no_measurement(fake_sysfs, monkeypatch):
    """With no threads available, reads report that they cannot measure."""
    fake_sysfs(vram_total_mb=4096, vis_total_mb=4096, vis_used_mb=0)

    def cannot_start(*args: object, **kwargs: object) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", cannot_start, raising=False)
    coordinator = VRAMCoordinator()  # must not raise
    assert coordinator._total_vram_mb == 8192, "the device value was not readable"
    assert coordinator._vram_detected_from_sysfs is False, "nothing was measured"

    result = coordinator.acquire(_make_spec(estimated_mb=512))
    assert result.granted is False
    assert result.fallback_to_cpu is True
    assert "unavailable" in result.reason


def test_contended_read_does_not_get_stuck_when_threads_cannot_start(fake_sysfs, monkeypatch):
    """A read that cannot start leaves the rate-limit flag clear."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=1024)
    coordinator = VRAMCoordinator(allow_contended_read=True)

    def cannot_start(*args: object, **kwargs: object) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", cannot_start, raising=False)
    assert coordinator._get_used_vram_contended() is None
    assert coordinator._contended_read_inflight is False, "the flag must not stay claimed"


def test_force_release_closes_instances(fake_sysfs):
    """The unhealthy-GPU path must return VRAM, not leak models."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()

    def factory(s: ModelSpec, n: int) -> FakeModel:
        return FakeModel()

    spec = _make_spec(estimated_mb=512)
    model, result = service.get_or_create(spec, factory)
    assert result.granted
    assert coordinator._allocations[spec.model_id].ref_count == 1

    with patch.object(coordinator, "_health_check", return_value=False):
        coordinator._handle_unhealthy_gpu()

    assert len(coordinator._allocations) == 0
    assert service._instances == {}
    assert model.closed is True
    assert coordinator._reservations == {}


def test_shutdown_closes_every_instance():
    """shutdown() closes each cached model once."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    with patch.object(coordinator, "_free_mb_from_used", return_value=65536):
        spec = _make_spec(estimated_mb=512)
        model, _ = service.get_or_create(spec, lambda s, n: FakeModel())
    service.shutdown()
    assert model.closed is True
    assert service._instances == {}


def test_close_failure_is_logged_not_swallowed(fake_sysfs, caplog):
    """A model whose close() raises must be reported, not silently ignored."""
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()

    class Broken(FakeModel):
        def close(self) -> None:
            raise RuntimeError("driver said no")

    bad = Broken()
    service._instances["/tmp/broken.gguf"] = bad
    with caplog.at_level("ERROR"):
        service.shutdown()
    assert "vram_instance_close_failed" in caplog.text


# --- Tests: Watchdog + deadlock detection ---


def test_watchdog_detects_stuck_operations(fake_sysfs):
    """An operation inflight beyond the timeout marks the GPU unhealthy."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    coordinator.track_operation_start("stuck_op_1")
    with coordinator._lock:
        coordinator._inflight["stuck_op_1"] = time.monotonic() - 5.0

    with patch.object(coordinator, "_handle_unhealthy_gpu") as mock_handle:
        coordinator._run_watchdog_iteration()
        assert mock_handle.called, "a stuck operation must trigger recovery"

    coordinator.track_operation_end("stuck_op_1")
    assert "stuck_op_1" not in coordinator._inflight


def test_stuck_operation_triggers_recovery_only_once(fake_sysfs):
    """A lost operation is reported once, not on every following tick.

    The operation stops being tracked when it is handled, so later ticks leave
    the models loaded in the meantime alone.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    coordinator.track_operation_start("lost_op")
    with coordinator._lock:
        coordinator._inflight["lost_op"] = time.monotonic() - 10.0

    with patch.object(coordinator, "_handle_unhealthy_gpu") as recovery:
        for _ in range(3):
            coordinator._run_watchdog_iteration()

    assert recovery.call_count == 1
    assert "lost_op" not in coordinator._inflight


def test_watchdog_does_not_close_models_loaded_after_recovery(fake_sysfs, monkeypatch):
    """Recovery for a lost operation must not keep closing healthy models."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    monkeypatch.setattr(vram_coordinator, "DEADLOCK_DETECTION_TIMEOUT_S", 0.01)
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()

    coordinator.track_operation_start("lost_op")
    with coordinator._lock:
        coordinator._inflight["lost_op"] = time.monotonic() - 10.0
    coordinator._run_watchdog_iteration()  # first tick recovers

    spec = _make_spec(estimated_mb=512)
    model, result = service.get_or_create(spec, lambda s, n: FakeModel())
    assert result.granted is True

    coordinator._run_watchdog_iteration()  # must not disturb the new model

    assert model.closed is False
    assert coordinator._allocations[spec.model_id].ref_count == 1
    assert result.reservation_id in coordinator._reservations


def test_health_check_unknown_on_host_without_amdgpu(fake_sysfs):
    """UNKNOWN is not UNHEALTHY: a non-AMD host is not a failing GPU."""
    fake_sysfs(vram_total_mb=None, vis_total_mb=None, vis_used_mb=None)
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_get_used_vram_lockfree", return_value=None):
        assert coordinator._health_check() is True
        assert coordinator._health == GPUHealth.UNKNOWN


def test_health_degrades_when_amdgpu_stops_answering(fake_sysfs):
    """On a host that does expose amdgpu, a failing read IS a failure."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_get_used_vram_lockfree", return_value=None):
        assert coordinator._health_check() is False
        assert coordinator._health == GPUHealth.DEGRADED


def test_health_transitions_to_healthy_after_lockfree_read():
    """After a successful lockfree read, health transitions to HEALTHY."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_get_used_vram_lockfree", return_value=2048):
        assert coordinator._health_check() is True
        assert coordinator._health == GPUHealth.HEALTHY


def test_initial_health_is_unknown():
    """On creation, health is UNKNOWN until the first health check."""
    coordinator = VRAMCoordinator()
    assert coordinator._health == GPUHealth.UNKNOWN


def test_watchdog_increments_health_failures():
    """If health_check fails consecutively, the GPU is marked UNHEALTHY."""
    coordinator = VRAMCoordinator()
    with (
        patch.object(coordinator, "_get_used_vram_lockfree", return_value=None),
        patch.object(coordinator, "_vram_detected_from_sysfs", True),
        patch.object(coordinator, "_handle_unhealthy_gpu") as mock_handle,
        patch.object(coordinator, "_health_check", return_value=False),
    ):
        for _ in range(HEALTH_CHECK_FAILURE_THRESHOLD + 1):
            coordinator._run_watchdog_iteration()
        assert mock_handle.called


def test_health_failure_threshold_reached_without_mocks(fake_sysfs):
    """The threshold is reachable in production code, not only under a mock."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_get_used_vram_lockfree", return_value=None):
        for _ in range(HEALTH_CHECK_FAILURE_THRESHOLD - 1):
            coordinator._run_watchdog_iteration()
            assert coordinator._health == GPUHealth.DEGRADED
        assert coordinator._consecutive_health_failures == HEALTH_CHECK_FAILURE_THRESHOLD - 1
    # A successful read clears the degradation
    with patch.object(coordinator, "_get_used_vram_lockfree", return_value=128):
        coordinator._run_watchdog_iteration()
        assert coordinator._health == GPUHealth.HEALTHY
        assert coordinator._consecutive_health_failures == 0


def test_recovery_requires_a_successful_read():
    """Recovery is declared only when the device answers again.

    A failed recovery leaves the GPU unhealthy, so new reservations keep
    falling back to CPU rather than being granted for a device that is not
    responding.
    """
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_health_check", return_value=False):
        coordinator._handle_unhealthy_gpu()
    assert coordinator._health == GPUHealth.UNHEALTHY

    with patch.object(coordinator, "_health_check", return_value=True):
        coordinator._handle_unhealthy_gpu()
    assert coordinator._health == GPUHealth.HEALTHY


def test_handle_unhealthy_gpu_clears_allocations():
    """If the GPU is unhealthy, all allocations are freed."""
    coordinator = VRAMCoordinator()
    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        result = coordinator.acquire(_make_spec(estimated_mb=512))
        assert result.granted
        with patch.object(coordinator, "_health_check", return_value=False):
            coordinator._handle_unhealthy_gpu()
        assert len(coordinator._allocations) == 0


def test_acquire_denies_when_unhealthy():
    """If the GPU is UNHEALTHY (previous deadlock), deny new acquisitions."""
    coordinator = VRAMCoordinator()
    coordinator._health = GPUHealth.UNHEALTHY
    result = coordinator.acquire(_make_spec(estimated_mb=512))
    assert result.granted is False
    assert "unhealthy" in result.reason.lower() or "deadlock" in result.reason.lower()


def test_watchdog_evicts_only_when_budget_is_tight(fake_sysfs):
    """Periodic eviction triggers on real pressure, and is skipped when unknown."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    with patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0):
        service = coordinator.get_embedding_service()
        spec = _make_spec(estimated_mb=512)
        model, result = service.get_or_create(spec, lambda s, n: FakeModel())
        coordinator.release(result.reservation_id)

        # Plenty of budget: nothing to evict
        coordinator._run_watchdog_iteration()
        assert coordinator._evictions == 0

        # Simulate a nearly full device: eviction must kick in
        with patch.object(coordinator, "_get_free_vram_mb", return_value=100):
            coordinator._run_watchdog_iteration()
        assert coordinator._evictions == 1
        assert model.closed is True

        # Unmeasurable: "unknown" is not "exhausted"
        with patch.object(coordinator, "_get_free_vram_mb", return_value=None):
            coordinator._run_watchdog_iteration()
        assert coordinator._evictions == 1


def test_stop_watchdog_waits_for_the_thread():
    """Stopping the watchdog must not claim success while the thread lives."""
    coordinator = VRAMCoordinator()
    coordinator.start_watchdog()
    thread = coordinator._watchdog_thread
    assert thread is not None and thread.is_alive()
    coordinator.stop_watchdog()
    assert not thread.is_alive()
    assert coordinator._watchdog_thread is None


# --- Tests: Concurrency ---


def test_budget_is_recomputed_under_the_lock(fake_sysfs):
    """Two callers that measure at the same time cannot both over-commit.

    The budget is recomputed under the lock that grants it, so a grant made
    while another caller was measuring is accounted for.
    """
    fake_sysfs(vram_total_mb=4096, vis_total_mb=4096, vis_used_mb=0)
    coordinator = VRAMCoordinator()  # free = 4096 - 1024 - 256 = 2816 MB
    measured_together = threading.Barrier(2, timeout=5.0)
    real_measure = coordinator._measure_used_mb

    def measure_then_wait() -> int | None:
        value = real_measure()
        # Both threads finish measuring before either one grants.
        measured_together.wait()
        return value

    results: list[AcquisitionResult] = []
    results_lock = threading.Lock()

    def worker(index: int) -> None:
        spec = _make_spec(model_id=f"/tmp/race-{index}.gguf", estimated_mb=2048, n_ctx=2048)
        result = coordinator.acquire(spec)
        with results_lock:
            results.append(result)

    with patch.object(coordinator, "_measure_used_mb", side_effect=measure_then_wait):
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

    granted = [result for result in results if result.granted]
    assert len(granted) == 1, "only one 2049MB request fits in a 2816MB budget"
    metrics = coordinator.get_metrics()
    assert metrics["vram_reserved_mb"] == 2049
    assert metrics["allocations"] == 1


def test_contended_read_is_rate_limited_under_concurrency(fake_sysfs, monkeypatch):
    """Simultaneous callers must not turn the rate limit into a burst of reads."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=256, vis_used_mb=233, vram_used_mb=1024)
    reads: list[str] = []
    real_read = vram_coordinator._read_sysfs_int

    def _spy(path_str: str, timeout_s: float) -> int | None:
        reads.append(path_str)
        return real_read(path_str, timeout_s)

    monkeypatch.setattr(vram_coordinator, "_read_sysfs_int", _spy)
    coordinator = VRAMCoordinator(allow_contended_read=True)

    def worker() -> None:
        coordinator._get_used_vram_contended()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    contended_reads = [path for path in reads if path.endswith("mem_info_vram_used")]
    assert len(contended_reads) == 1


def test_eviction_does_not_race_with_acquisition(fake_sysfs):
    """Eviction and acquisition mutate the allocation table concurrently.

    Eviction takes the coordinator lock, so it cannot corrupt an iteration in
    progress or leave partially applied state behind.
    """
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker() -> None:
        index = 0
        try:
            while not stop.is_set():
                spec = _make_spec(
                    model_id=f"/tmp/race-{threading.get_ident()}-{index}.gguf",
                    estimated_mb=64,
                )
                result = coordinator.acquire(spec)
                index += 1
                if result.granted:
                    coordinator.release(result.reservation_id)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    def evictor() -> None:
        try:
            while not stop.is_set():
                coordinator._evict_idle(1)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    with patch.object(vram_coordinator, "IDLE_EVICTION_TTL_S", 0.0):
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
        threads.append(threading.Thread(target=evictor, daemon=True))
        for thread in threads:
            thread.start()
        time.sleep(1.0)
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)

    assert not errors, f"concurrent eviction raised: {errors!r}"


def test_no_deadlock_under_concurrent_acquisition():
    """Four concurrent workers loading the same model share one instance.

    This is the scenario reported as causing the original lockup: four
    workers instantiating the same GGUF model concurrently. With the
    coordinator, the factory runs once and the remaining callers receive the
    shared instance, so only one model is loaded.
    """
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    call_count = {"count": 0}
    call_lock = threading.Lock()
    sentinel = object()  # unique sentinel for identity comparison

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> object:
        with call_lock:
            call_count["count"] += 1
        time.sleep(0.01)  # simulate loading
        return sentinel

    with patch.object(coordinator, "_free_mb_from_used", return_value=4096):
        spec = _make_spec(estimated_mb=512)
        results: list[tuple[object, object]] = []
        results_lock = threading.Lock()

        def worker() -> None:
            instance, result = service.get_or_create(spec, factory)
            with results_lock:
                results.append((instance, result))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "Thread hung — concurrency deadlock detected"

    instances = {result[0] for result in results if result[0] is not None}
    assert len(instances) == 1, f"Expected 1 unique instance, got {len(instances)}"
    assert sentinel in instances
    assert call_count["count"] == 1, "the factory must run exactly once"
    for _instance, result in results:
        assert getattr(result, "granted", False) is True
    assert coordinator._allocations[spec.model_id].ref_count == 4


# --- Tests: input validation ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "", "estimated_mb": 1},
        {"model_id": "/tmp/x.gguf", "estimated_mb": 0},
        {"model_id": "/tmp/x.gguf", "estimated_mb": -1},
        {"model_id": "/tmp/x.gguf", "estimated_mb": 1, "n_ctx": 0},
        {"model_id": "/tmp/x.gguf", "estimated_mb": 1, "n_gpu_layers": -2},
        {"model_id": "/tmp/x.gguf", "estimated_mb": 1, "total_layers": 0},
    ],
)
def test_model_spec_rejects_invalid_values(kwargs):
    """Bad specs fail loudly at construction instead of mid-load."""
    with pytest.raises(ValueError):
        ModelSpec(**kwargs)


# --- Tests: Metrics ---


def test_metrics_track_allocations(fake_sysfs):
    """Metrics reflect acquisitions, reservations and health."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    assert coordinator.acquire(_make_spec(estimated_mb=512)).granted
    metrics = coordinator.get_metrics()
    assert metrics["acquisitions_total"] == 1
    assert metrics["acquisitions_denied"] == 0
    assert metrics["allocations"] == 1
    assert metrics["ref_count_total"] == 1
    assert metrics["vram_total_mb"] == 8192
    assert metrics["vram_reserved_mb"] == 513  # 512 MB + 1 MB KV cache
    assert metrics["vram_measurement_source"] == "lockfree_visible_window"
    assert metrics["visible_vram_mb"] == 8192
    assert metrics["releases"] == 0
    assert metrics["evictions"] == 0
    assert metrics["inflight_operations"] == 0
    assert metrics["health"] == "unknown"


# --- Tests: Singleton ---


def test_get_coordinator_returns_singleton():
    """get_coordinator always returns the same instance."""
    from vram_coordinator import get_coordinator

    assert get_coordinator() is get_coordinator()


def test_reset_coordinator_clears_singleton():
    """reset_coordinator clears the singleton (test use)."""
    from vram_coordinator import get_coordinator

    first = get_coordinator()
    reset_coordinator()
    assert get_coordinator() is not first


def test_shutdown_coordinator_stops_the_watchdog():
    """shutdown_coordinator leaves no watchdog thread behind.

    The thread reference is cleared only once the thread has exited, so a
    timed-out stop cannot be mistaken for a clean one.
    """
    from vram_coordinator import get_coordinator

    coordinator = get_coordinator()
    thread = coordinator._watchdog_thread
    assert thread is not None and thread.is_alive()
    shutdown_coordinator()
    assert not thread.is_alive()
    assert coordinator._watchdog_thread is None


# --- Tests: VRAM reservation context manager ---


def test_reserve_context_manager_grants_and_releases(fake_sysfs):
    """with coordinator.reserve(spec) as result: ... releases on exit."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    with coordinator.reserve(spec) as result:
        assert result.granted
        assert spec.model_id in coordinator._allocations
        assert coordinator._allocations[spec.model_id].ref_count == 1
    # On exit the refcount drops, but the allocation stays until eviction
    assert spec.model_id in coordinator._allocations
    assert coordinator._allocations[spec.model_id].ref_count == 0


def test_reserve_context_manager_releases_on_exception(fake_sysfs):
    """If an exception occurs inside the with, the reservation is still released."""
    fake_sysfs(vram_total_mb=8192, vis_total_mb=8192, vis_used_mb=0)
    coordinator = VRAMCoordinator()
    spec = _make_spec(estimated_mb=512)
    with contextlib.suppress(RuntimeError), coordinator.reserve(spec) as result:
        assert result.granted
        raise RuntimeError("test exception")
    assert coordinator._allocations[spec.model_id].ref_count == 0


# --- Tests: reported incident — concurrent loads of a shared embedding model ---


def test_shared_embedding_model_is_offloaded_within_budget():
    """A small model that fits is offloaded fully, and shared by all callers.

    The model used in the reported incident was a 24 MB embedding model
    requested with ``n_gpu_layers=-1``. It fits well inside the budget, so
    full offload is the correct outcome; what changes is that the request is
    now validated and that concurrent callers share one instance.
    """
    coordinator = VRAMCoordinator()
    service = coordinator.get_embedding_service()
    layers_seen: list[int] = []

    def factory(spec: ModelSpec, n_gpu_layers_actual: int) -> int:
        layers_seen.append(n_gpu_layers_actual)
        return id(spec)

    with patch.object(coordinator, "_free_mb_from_used", return_value=80):
        spec = _make_spec(
            model_id="bge-small-24MB",
            estimated_mb=24,
            n_ctx=4096,
            n_gpu_layers=-1,  # the value that triggered the incident
        )
        instance, result = service.get_or_create(spec, factory)
        assert result.granted is True
        assert instance is not None
        assert layers_seen == [-1], "24MB fits in 80MB, so full offload is correct"
        assert coordinator._allocations[spec.model_id].reserved_mb == 26  # 24 MB + 2 MB KV
    service.release(spec.model_id)
