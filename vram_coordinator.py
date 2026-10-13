"""
VRAM Coordinator — single source of truth for the GPU memory budget.

This module coordinates concurrent GPU model loads on AMD hardware. It was
written after a system running four llama-cpp workers, each loading the same
GGUF embedding model with ``n_gpu_layers=-1``, rebooted following a kernel
soft lockup traced to GPU memory accounting.

INVARIANTS:
    - **Single source of truth** for the VRAM budget: every granted
      reservation is subtracted from subsequent budget computations, so
      concurrent loads cannot over-commit the device.
    - **Reserve before load**. Free VRAM is never assumed to be available.
    - **Reference counting** for shared models: two callers requesting the
      same ``model_id`` share one instance.
    - **Graceful eviction**: an unreferenced model becomes eviction-eligible
      after ``IDLE_EVICTION_TTL_S``, and eviction closes the instance so its
      memory is returned to the driver.
    - **Watchdog**: each tick checks that the device answers, treats an
      operation that does not complete within ``DEADLOCK_DETECTION_TIMEOUT_S``
      as lost, and evicts idle models when the budget is tight.
    - **Explicit layer semantics**: ``n_gpu_layers=-1`` requests all layers
      and is denied when the model does not fit; it is never reduced
      implicitly. An explicit ``n_gpu_layers >= 0`` is passed through
      unchanged.
    - **No hidden performance loss**: when VRAM is available, models run on
      the GPU. When it is not, the caller is told to fall back to CPU.
      ``n_ctx`` is never modified.

VRAM MEASUREMENT:
    ``mem_info_vis_vram_used`` is the only used-memory attribute that takes
    no driver lock, but the kernel defines it as the amount of *visible*
    VRAM in use (``amdgpu_vram_mgr_vis_usage``, bounded by
    ``visible_vram_size``, the BAR aperture). On a device without a large
    BAR the value covers only a fraction of memory: an RX 580 exposes a
    256 MB window on an 8 GB device, so a workload using 5 GB reports about
    233 MB. A budget derived from that value would be optimistic, so the
    lock-free read is trusted only when the visible window covers at least
    ``VISIBLE_WINDOW_MIN_RATIO`` of total VRAM. Otherwise the coordinator
    reports that it cannot measure and the caller falls back to CPU, unless
    the operator explicitly enables ``allow_contended_read=True``, which
    reads the authoritative counter under a rate limit.

    The authoritative attribute is ``mem_info_vram_used``. It is derived
    from ``ttm_resource_manager_usage()``, which takes the TTM device
    ``lru_lock`` spinlock — the same lock the driver takes for every VRAM
    allocation and free.

RECURRENCE PREVENTION:
    - Reported failure: four concurrent ``Llama(n_gpu_layers=-1)`` loads
      while ksystemstats sampled VRAM. The kernel logged a soft lockup with
      a task stuck for 1717 s, and the machine rebooted.
    - Mitigation: ``EmbeddingService`` coalesces the four loads into one
      model with four references, the budget accounts for every reservation
      the coordinator itself granted, and the contended
      ``mem_info_vram_used`` path is not read unless explicitly allowed.
"""

from __future__ import annotations

import enum
import glob
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

logger = logging.getLogger("VRAMCoordinator")

# --- Constants ---

#: Fixed reserve for ksystemstats, kwin_wayland, and other system processes.
#: 1 GB on an 8 GB GPU is the minimum the Wayland + kwin stack requires
#: during Vulkan transitions. Reduces the available budget from 8 to 7 GB.
SYSTEM_VRAM_RESERVE_MB: int = 1024

#: Dynamic headroom before declaring "VRAM exhausted". If free_mb is below
#: this, no new reservation is allowed. 256 MB covers the transient spike
#: of allocations.
SAFETY_HEADROOM_MB: int = 256

#: Critical threshold that triggers automatic eviction. 90% of the budget.
#: Above this, the LRU is freed.
EVICTION_THRESHOLD_PCT: float = 0.90

#: Timeout in seconds to detect the amdgpu deadlock pattern. If a GPU
#: operation does not complete within this time, the watchdog releases it.
DEADLOCK_DETECTION_TIMEOUT_S: float = 2.0

#: TTL of idle models before they become eligible for eviction.
IDLE_EVICTION_TTL_S: float = 60.0

#: Number of consecutive failed health checks before marking the GPU unhealthy.
HEALTH_CHECK_FAILURE_THRESHOLD: int = 3

#: Watchdog interval in seconds.
WATCHDOG_INTERVAL_S: float = 5.0

#: The lock-free ``mem_info_vis_vram_used`` read is only a valid proxy for
#: total VRAM usage when the visible window covers (almost) the whole device.
#: Below this ratio the measurement is treated as unavailable, because the
#: visible window is a BAR aperture and under-reports real usage.
VISIBLE_WINDOW_MIN_RATIO: float = 0.95

#: Upper bound for a single sysfs read. A stalled driver would otherwise
#: block the watchdog on the very path it monitors.
SYSFS_READ_TIMEOUT_S: float = 1.0

#: Minimum seconds between two reads of the contended
#: ``mem_info_vram_used`` path when the operator opts in with
#: ``allow_contended_read=True``. Keeps the coordinator's own contribution
#: to the lock contention negligible.
CONTENDED_READ_MIN_INTERVAL_S: float = 10.0

#: Glob matching the amdgpu sysfs directory of every DRM device that exposes VRAM
#: attributes. The first match is resolved once and used for all readings, so
#: the totals and the usage always come from the same device on a multi-GPU
#: host.
_DRM_DEVICE_GLOB: str = "/sys/class/drm/card*/device"

#: Total size of the device VRAM. Static, takes no driver lock.
_ATTR_VRAM_TOTAL: str = "mem_info_vram_total"

#: Size of the CPU-visible VRAM window (BAR aperture). Static, no driver lock.
_ATTR_VIS_VRAM_TOTAL: str = "mem_info_vis_vram_total"

#: VRAM used inside the visible window. Lock-free, but window-scoped.
_ATTR_VIS_VRAM_USED: str = "mem_info_vis_vram_used"

#: Total VRAM used. Authoritative, but guarded by the TTM lru_lock spinlock.
_ATTR_VRAM_USED: str = "mem_info_vram_used"


__all__ = [
    "CONTENDED_READ_MIN_INTERVAL_S",
    "DEADLOCK_DETECTION_TIMEOUT_S",
    "EVICTION_THRESHOLD_PCT",
    "HEALTH_CHECK_FAILURE_THRESHOLD",
    "IDLE_EVICTION_TTL_S",
    "SAFETY_HEADROOM_MB",
    "SYSFS_READ_TIMEOUT_S",
    "SYSTEM_VRAM_RESERVE_MB",
    "VISIBLE_WINDOW_MIN_RATIO",
    "WATCHDOG_INTERVAL_S",
    "AcquisitionResult",
    "EmbeddingService",
    "GPUHealth",
    "ModelAllocation",
    "ModelSpec",
    "VRAMCoordinator",
    "VRAMReservation",
    "get_coordinator",
    "reset_coordinator",
    "shutdown_coordinator",
]

#: Keep in sync with the ``version`` field in pyproject.toml.
__version__ = "0.1.0"


# --- sysfs read helper ---


class _SharedRead:
    """A sysfs read in progress, shareable by concurrent callers."""

    __slots__ = ("done", "value")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: int | None = None


#: Reads in progress, one per path, and the paths whose timeout was reported.
#: Concurrent callers share the read that is already running instead of
#: starting another one, so a device that stops responding leaves a single
#: blocked thread per path rather than one per call.
_reads: dict[str, _SharedRead] = {}
_reads_lock = threading.Lock()
_timeout_reported: set[str] = set()


def _warn_stalled_read(path_str: str, timeout_s: float) -> None:
    """Report a read that overran its timeout, once per path."""
    with _reads_lock:
        if path_str in _timeout_reported:
            return
        _timeout_reported.add(path_str)
    logger.warning(
        "vram_sysfs_read_timeout",
        extra={"path": path_str, "timeout_s": timeout_s},
    )


def _read_sysfs_int(path_str: str, timeout_s: float) -> int | None:
    """Read an integer from a sysfs file, giving up after ``timeout_s``.

    The read runs in a daemon thread because a sysfs read can block
    indefinitely when the GPU is unresponsive, and the watchdog must not block
    on the path it monitors.

    One read at a time is kept per path: concurrent callers wait for the read
    already in progress and share its result, and a reader that overran its
    timeout keeps the slot until it finishes, so a stalled device cannot
    accumulate a thread per call. Returns None on error, stall or timeout.
    """
    with _reads_lock:
        shared = _reads.get(path_str)
        if shared is None:
            shared = _SharedRead()
            _reads[path_str] = shared
            starter = True
        else:
            starter = False

    if not starter:
        if not shared.done.wait(timeout=timeout_s):
            return None
        return shared.value

    def _read() -> None:
        try:
            raw = Path(path_str).read_text(encoding="utf-8").strip()
            shared.value = int(float(raw))
        except (OSError, ValueError):
            shared.value = None
        finally:
            with _reads_lock:
                _reads.pop(path_str, None)
                _timeout_reported.discard(path_str)
            shared.done.set()

    reader = threading.Thread(target=_read, name="VRAMSysfsRead", daemon=True)
    try:
        reader.start()
    except RuntimeError:
        # The process cannot create more threads: report that the value could
        # not be read so the caller falls back to CPU instead of raising.
        with _reads_lock:
            _reads.pop(path_str, None)
        shared.done.set()
        return None
    if not shared.done.wait(timeout=timeout_s):
        _warn_stalled_read(path_str, timeout_s)
        return None
    return shared.value


# --- Types ---


class GPUHealth(enum.Enum):
    """GPU health state according to the watchdog."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # some health checks failed
    UNHEALTHY = "unhealthy"  # probable deadlock, reject new reservations
    UNKNOWN = "unknown"  # sysfs unavailable


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a model for VRAM reservation.

    Attributes:
        model_id: Unique identifier (absolute path to the GGUF or logical name).
        estimated_mb: Estimated VRAM of the model fully loaded on GPU.
        n_ctx: Context size. Deducted from the available VRAM budget.
        n_gpu_layers: GPU layers. ``-1`` requests every layer: the request is
            denied when the whole model does not fit and is never reduced
            implicitly. Any explicit value ``>= 0`` is handed to the loader
            unchanged and honoured as-is.
        total_layers: Total number of layers in the model. Used to turn an
            explicit ``n_gpu_layers`` into a fraction of ``estimated_mb``.
            When it is unknown, a partial offload is budgeted as the whole
            model, which avoids an under-estimate.
    """

    model_id: str
    estimated_mb: int
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    total_layers: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if self.estimated_mb <= 0:
            raise ValueError(f"estimated_mb must be > 0, got {self.estimated_mb}")
        if self.n_ctx <= 0:
            raise ValueError(f"n_ctx must be > 0, got {self.n_ctx}")
        if self.n_gpu_layers < -1:
            raise ValueError(
                f"n_gpu_layers must be -1 (all layers) or >= 0, got {self.n_gpu_layers}"
            )
        if self.total_layers is not None and self.total_layers <= 0:
            raise ValueError(f"total_layers must be > 0 when given, got {self.total_layers}")


@dataclass
class ModelAllocation:
    """Record of a model loaded in VRAM.

    Attributes:
        spec: Model specification.
        ref_count: Number of current holders.
        last_used: Timestamp of last use (for LRU eviction).
        gpu_layers_actual: Layers handed to the loader when the model was
            loaded (``-1`` means every layer).
        reserved_mb: VRAM this allocation was granted against the budget.
            Subtracted from every later budget computation, so concurrent
            loads cannot over-commit the device.
    """

    spec: ModelSpec
    ref_count: int = 0
    last_used: float = field(default_factory=time.monotonic)
    gpu_layers_actual: int = 0
    reserved_mb: int = 0


@dataclass
class AcquisitionResult:
    """Result of a VRAM budget acquisition.

    Attributes:
        granted: True if the reservation was granted.
        reservation_id: Handle to pass to ``release`` (``None`` if not granted).
        reason: Why the request was denied, if it was.
        gpu_layers_actual: Layers to hand to the loader (``-1`` means every
            layer).
        fallback_to_cpu: True if the caller must load on CPU instead.
        already_loaded: True when another holder already has this model loaded
            and the reservation was shared. The caller must not load another
            copy: the existing instance has to be reused, which is what
            ``EmbeddingService.get_or_create`` does. This flag is False for
            every grant that corresponds to a model the caller loads itself.
    """

    granted: bool
    reservation_id: str | None = None
    reason: str = ""
    gpu_layers_actual: int = 0
    fallback_to_cpu: bool = False
    already_loaded: bool = False


class VRAMReservation:
    """Context manager that holds a reservation for the duration of a block.

    Usage::

        with coordinator.reserve(ModelSpec(...)) as result:
            if result.granted:
                ...  # load on GPU
            else:
                ...  # load on CPU

    The reservation is released when the block exits, including on exception.
    """

    def __init__(self, coordinator: VRAMCoordinator, spec: ModelSpec) -> None:
        self._coordinator = coordinator
        self._spec = spec
        self._result: AcquisitionResult | None = None
        self._released = False

    def __enter__(self) -> AcquisitionResult:
        self._result = self._coordinator.acquire(self._spec)
        return self._result

    def __exit__(self, *args: object) -> None:
        if not self._released and self._result and self._result.granted:
            self._coordinator.release(self._result.reservation_id)  # type: ignore[arg-type]
        self._released = True


class EmbeddingService:
    """Shared model service with reference counting.

    Callers that need the same model share one instance per ``model_id``
    instead of loading a private copy each, which bounds the VRAM a shared
    model such as an embedding model consumes.

    This service is the sole owner of the ``_instances`` cache; the
    coordinator never mutates it. Every path that drops an instance goes
    through :meth:`_discard`, which closes the model so its memory is
    returned to the driver.
    """

    def __init__(self, coordinator: VRAMCoordinator) -> None:
        self._coordinator = coordinator
        self._instances: dict[str, Any] = {}
        # Reentrant: get_or_create holds it across the VRAM acquisition, and
        # that path can evict an idle model, which discards and closes an
        # instance through this same lock on the same thread.
        self._lock = threading.RLock()

    def get_or_create(
        self,
        spec: ModelSpec,
        factory: Callable[[ModelSpec, int], Any],
    ) -> tuple[Any, AcquisitionResult]:
        """Get the shared model instance, creating it if it does not exist.

        Concurrency: the lock is held for the whole creation sequence
        (lookup, VRAM acquisition, factory call and registration), so
        concurrent callers cannot create duplicate instances for the same
        ``model_id``.

        Lifecycle: the service owns the lifecycle. The first call acquires
        VRAM (ref_count 1) and keeps it until ``release(model_id)`` drops
        the last reference. Subsequent calls only increment the shared
        reference count and do not acquire again. This is the opposite of
        the :meth:`VRAMCoordinator.reserve` context manager, which releases
        when the block exits.

        Args:
            spec: Model specification.
            factory: Callable that creates the model. Receives
                ``(spec, n_gpu_layers_actual)`` and returns the instance.

        Returns:
            Tuple ``(instance, acquisition_result)``. ``instance`` is
            ``None`` when the acquisition or the factory failed.
        """
        with self._lock:
            existing = self._instances.get(spec.model_id)
            if existing is not None:
                # Registering the extra holder is atomic with checking that the
                # model is still allocated, so an eviction cannot slip in
                # between and leave the caller with an instance that is about
                # to be closed.
                grant = self._coordinator._register_shared_grant(spec.model_id)
                if grant is not None:
                    handle, gpu_layers_actual = grant
                    return existing, AcquisitionResult(
                        granted=True,
                        reservation_id=handle,
                        gpu_layers_actual=gpu_layers_actual,
                        already_loaded=True,
                    )
                # The instance outlived its allocation (eviction, force-release
                # or a failed reload): drop it and load a fresh one below
                # instead of handing out a model the budget no longer tracks.
                self._instances.pop(spec.model_id, None)
                self._close_instance(spec.model_id, existing)

            # Not cached: acquire VRAM and load. The lock is held for the
            # whole sequence so other callers observe the instance as soon
            # as it is registered. The reserve() context manager is not used
            # here because it releases on block exit, which would drop the
            # reservation as soon as this method returned; the service keeps
            # the reference instead and releases it from release(model_id).
            acq_result = self._coordinator.acquire(spec)
            if not acq_result.granted:
                return None, acq_result
            if acq_result.already_loaded:
                # The model is loaded by a holder outside this service, so
                # there is no instance to share. Reporting it and giving the
                # reference back keeps the caller from loading a second copy,
                # which is the outcome the budget exists to prevent.
                self._coordinator.release(acq_result.reservation_id)  # type: ignore[arg-type]
                return None, AcquisitionResult(
                    granted=False,
                    reason="model_loaded_outside_the_service",
                    fallback_to_cpu=True,
                )
            try:
                instance = factory(spec, acq_result.gpu_layers_actual)
            except BaseException:
                # The loader raised: nothing was loaded, so the reservation and
                # its handle are returned, including when the loader fails with
                # an error instead of returning None.
                self._coordinator._abandon_allocation(spec.model_id, acq_result.reservation_id)
                raise
            if instance is None:
                # The factory returned no model: the budget is returned
                # immediately instead of waiting for the eviction TTL.
                self._coordinator._abandon_allocation(spec.model_id, acq_result.reservation_id)
                return None, AcquisitionResult(
                    granted=False,
                    reason="factory_returned_none",
                    fallback_to_cpu=True,
                )
            # Registered inside self._lock so other callers see it at once.
            self._instances[spec.model_id] = instance
            return instance, acq_result

    def release(self, model_id: str) -> None:
        """Release one reference held by this service.

        Dropping the last reference discards (and closes) the cached instance,
        so the next ``get_or_create`` for the same model loads it again instead
        of returning a model the budget no longer accounts for. Every other
        release revokes one reservation handle, so the number of handles keeps
        matching the number of live references.
        """
        with self._lock:
            allocation_released = self._coordinator._release_service_reference(model_id)
            stale = self._instances.pop(model_id, None) if allocation_released else None
        if stale is not None:
            self._close_instance(model_id, stale)

    def _discard(self, model_id: str) -> None:
        """Drop a cached instance whose allocation the coordinator dropped.

        Called by the coordinator after releasing its own lock, so the
        coordinator lock is never held while this service lock is taken. If the
        model has been acquired again in the meantime, the cached instance
        belongs to that new reservation and is left alone: closing it would take
        a model away from a caller that holds it.
        """
        with self._lock:
            if self._coordinator._is_allocated(model_id):
                return
            instance = self._instances.pop(model_id, None)
        if instance is not None:
            self._close_instance(model_id, instance)

    def _close_instance(self, model_id: str, instance: Any) -> None:
        """Close a model instance, returning its VRAM to the driver.

        Called with no lock held on the eviction, release and shutdown paths.
        The stale-instance path of :meth:`get_or_create` does hold this
        service's lock, because the caller must not receive an instance that is
        being closed; a slow ``close()`` therefore blocks other service callers
        on that path only.
        """
        close = getattr(instance, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            logger.exception(
                "vram_instance_close_failed",
                extra={"model_id": model_id},
            )

    def shutdown(self) -> None:
        """Shut down all instances."""
        with self._lock:
            instances = list(self._instances.items())
            self._instances.clear()
        for model_id, instance in instances:
            self._close_instance(model_id, instance)


# --- Coordinator ---


class VRAMCoordinator:
    """Central VRAM budget coordinator.

    Components:
        - **Sysfs reader**: reads VRAM without acquiring the driver spinlock.
        - **Budget manager**: reserves and frees budget per model.
        - **Reference counting**: multiple holders of the same model share it.
        - **Watchdog**: detects the amdgpu deadlock and force-releases.
        - **Health check**: verifies the GPU responds to new submissions.
    """

    def __init__(
        self,
        *,
        allow_contended_read: bool = False,
        contended_read_min_interval_s: float = CONTENDED_READ_MIN_INTERVAL_S,
    ) -> None:
        """
        Args:
            allow_contended_read: Enables reading ``mem_info_vram_used``
                when the lock-free visible-window read cannot be trusted.
                That attribute is guarded by the TTM ``lru_lock`` spinlock
                that ksystemstats also takes, so it stays disabled unless the
                operator accepts the trade-off. Reads remain rate-limited.
            contended_read_min_interval_s: Minimum seconds between two
                contended reads; the last value is cached in between.
        """
        # Lock for all internal operations (single-writer, multi-reader).
        self._lock = threading.RLock()
        self._allocations: dict[str, ModelAllocation] = {}
        # reservation_id -> model_id, so release() can validate its handle.
        self._reservations: dict[str, str] = {}
        # Source of unique reservation handles.
        self._handle_counter = count(1)
        # Cache of total VRAM (static, read once at startup).
        self._total_vram_mb: int | None = None
        # Directory of the resolved amdgpu device; every reading comes from it.
        self._device_dir: Path | None = None
        # True when total VRAM came from sysfs (an AMD host) instead of the
        # 8192 MB fallback. A failing read on such a host is a health signal.
        self._vram_detected_from_sysfs: bool = False
        # Size of the CPU-visible window and whether it covers the whole device.
        # Only then is the lock-free vis_vram_used read a valid usage proxy.
        self._visible_vram_mb: int | None = None
        self._visible_window_trustworthy: bool = False
        # Opt-in access to the contended mem_info_vram_used path.
        self._allow_contended_read = allow_contended_read
        self._contended_read_min_interval_s = contended_read_min_interval_s
        self._contended_read_lock = threading.Lock()
        self._contended_read_inflight: bool = False
        self._last_contended_read: float = 0.0
        self._contended_used_mb: int | None = None
        # Health tracking
        self._health: GPUHealth = GPUHealth.UNKNOWN
        self._consecutive_health_failures: int = 0
        # Watchdog
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop_event = threading.Event()
        self._watchdog_lock = threading.Lock()
        # Inflight operations tracking for deadlock detection
        self._inflight: dict[str, float] = {}  # op_id -> start_time
        # Embedding service
        self._embedding_service = EmbeddingService(self)
        # Metrics
        self._acquisitions_total: int = 0
        self._acquisitions_denied: int = 0
        self._evictions: int = 0
        self._releases: int = 0
        # Detect total VRAM (does not fail if sysfs is unavailable)
        self._detect_total_vram()

    # -- Public API ---------------------------------------------------------

    def reserve(self, spec: ModelSpec) -> VRAMReservation:
        """Reserve VRAM for a model. Context manager."""
        return VRAMReservation(self, spec)

    def acquire(self, spec: ModelSpec) -> AcquisitionResult:
        """Acquire VRAM budget to load a model.

        Algorithm:
            1. If the model is already loaded (ref_count > 0), register one
               more holder and return; this path reads no sysfs.
            2. Read the current measurement outside the lock, so sysfs I/O
               never serializes the rest of the API.
            3. If the request would not fit, evict idle models and close them
               outside the lock. Closing takes the service lock, and the
               service takes this lock, so the two must not nest.
            4. Under the lock, compute the available memory as
               total - used - system reserve - safety headroom - every
               reservation this coordinator already granted, so the decision
               and the reservations it depends on cannot drift apart.
            5. Deny when the GPU is unhealthy or when usage cannot be
               measured; the caller falls back to CPU in both cases.
            6. If all layers (-1) were requested and the model does not fit,
               deny the request instead of reducing it.
            7. Otherwise grant the request and register the reservation.

        Note:
            The budget is cumulative: ``acquire`` subtracts the reservations
            it already handed out, so N workers cannot each validate against
            the same free VRAM and over-commit the device.
        """
        # Fast path: an already loaded model needs no measurement, only one
        # more holder. Keeping it first avoids a sysfs read on the hot path
        # taken by every additional holder of a shared model.
        with self._lock:
            shared = self._register_shared_grant(spec.model_id)
            if shared is not None:
                self._acquisitions_total += 1
                return self._shared_grant_result(spec.model_id, *shared)

        # Read outside the lock: a stalled driver must not block release()
        # or the embedding service for other callers.
        used_mb = self._measure_used_mb()

        # Free idle models before deciding, outside the lock. They are closed
        # through the service, so this cannot run under the coordinator lock.
        with self._lock:
            free_mb = self._free_mb_from_used(used_mb)
            required_mb = self._estimate_vram_mb(spec, spec.n_gpu_layers)
            deficit_mb = 0 if free_mb is None else required_mb - free_mb
        if deficit_mb > 0:
            self._evict_idle(deficit_mb)

        with self._lock:
            self._acquisitions_total += 1

            # A concurrent caller may have loaded the model while the
            # measurement was taken or while idle models were being evicted.
            shared = self._register_shared_grant(spec.model_id)
            if shared is not None:
                return self._shared_grant_result(spec.model_id, *shared)

            # 2. Check health
            if self._health is GPUHealth.UNHEALTHY:
                self._acquisitions_denied += 1
                return AcquisitionResult(
                    granted=False,
                    reason="gpu_unhealthy_deadlock_suspected",
                    fallback_to_cpu=True,
                )

            # 3. The budget is recomputed here, with the reservations of every
            # grant that happened while the measurement was being read and with
            # the reservations released by the eviction above.
            free_mb = self._free_mb_from_used(used_mb)

            # 4. Usage cannot be measured: deny and let the caller use CPU
            if free_mb is None:
                self._acquisitions_denied += 1
                return AcquisitionResult(
                    granted=False,
                    reason=self._measurement_unavailable_reason(),
                    fallback_to_cpu=True,
                )

            # 5. A request for all layers (-1) that does not fit is denied.
            # The request is not converted into a partial offload: a caller
            # that wants partial GPU passes an explicit layer count, so -1
            # keeps a single unambiguous meaning.
            if spec.n_gpu_layers == -1 and spec.estimated_mb > free_mb:
                self._acquisitions_denied += 1
                return AcquisitionResult(
                    granted=False,
                    reason=(
                        f"model_too_large_for_full_gpu: "
                        f"estimated={spec.estimated_mb}MB, free={free_mb}MB; "
                        f"specify n_gpu_layers > 0 for partial GPU"
                    ),
                    fallback_to_cpu=True,
                )

            # 6. Layers handed to the loader. -1 means all layers and the
            # model is known to fit at this point; an explicit count is
            # passed through unchanged. Neither is modified implicitly.
            gpu_layers_actual = spec.n_gpu_layers
            required_mb = self._estimate_vram_mb(spec, gpu_layers_actual)
            if required_mb > free_mb:
                self._acquisitions_denied += 1
                return AcquisitionResult(
                    granted=False,
                    reason=(f"vram_exhausted: required={required_mb}MB, free={free_mb}MB"),
                    fallback_to_cpu=True,
                )

            # 7. Register allocation
            reservation_id = self._new_handle(spec.model_id)
            self._allocations[spec.model_id] = ModelAllocation(
                spec=spec,
                ref_count=1,
                last_used=time.monotonic(),
                gpu_layers_actual=gpu_layers_actual,
                reserved_mb=required_mb,
            )
            self._reservations[reservation_id] = spec.model_id
            logger.info(
                "vram_reservation_granted",
                extra={
                    "model_id": spec.model_id,
                    "reservation_id": reservation_id,
                    "gpu_layers_actual": gpu_layers_actual,
                    "vram_free_mb": free_mb,
                    "vram_required_mb": required_mb,
                    "vram_total_mb": self._total_vram_mb,
                    "gpu_health": self._health.value,
                },
            )
            return AcquisitionResult(
                granted=True,
                reservation_id=reservation_id,
                gpu_layers_actual=gpu_layers_actual,
            )

    def _shared_grant_result(
        self,
        model_id: str,
        reservation_id: str,
        gpu_layers_actual: int,
    ) -> AcquisitionResult:
        """Result for a grant that shares a model another holder loaded."""
        logger.debug(
            "vram_shared_grant",
            extra={
                "model_id": model_id,
                "reservation_id": reservation_id,
                "ref_count": self._allocations[model_id].ref_count,
            },
        )
        return AcquisitionResult(
            granted=True,
            reservation_id=reservation_id,
            gpu_layers_actual=gpu_layers_actual,
            already_loaded=True,
        )

    def release(self, reservation_id: str) -> None:
        """Release a reservation. Decrements the ref count.

        The handle is validated against the reservations actually granted, and
        every grant produces a distinct handle. An unknown or already-released
        handle is reported and ignored, so a stale or repeated release can
        never decrement a different, live allocation.
        """
        with self._lock:
            model_id = self._reservations.pop(reservation_id, None)
            if model_id is None:
                logger.warning(
                    "vram_release_unknown_handle",
                    extra={"reservation_id": reservation_id},
                )
                return
            self._releases += 1
            self._decrement_ref(model_id)

    def get_embedding_service(self) -> EmbeddingService:
        """Get the shared embedding service."""
        return self._embedding_service

    def get_health(self) -> GPUHealth:
        """Get the GPU health state."""
        return self._health

    def get_metrics(self) -> dict[str, int | float | str | None]:
        """Metrics for observability."""
        with self._lock:
            return {
                "allocations": len(self._allocations),
                "ref_count_total": sum(a.ref_count for a in self._allocations.values()),
                "vram_total_mb": self._total_vram_mb or 0,
                "visible_vram_mb": self._visible_vram_mb,
                "vram_measurement_source": self._measurement_source(),
                "vram_reserved_mb": sum(a.reserved_mb for a in self._allocations.values()),
                "acquisitions_total": self._acquisitions_total,
                "acquisitions_denied": self._acquisitions_denied,
                "releases": self._releases,
                "evictions": self._evictions,
                "inflight_operations": len(self._inflight),
                "health": self._health.value,
            }

    def start_watchdog(self) -> None:
        """Start the watchdog in a daemon thread."""
        with self._watchdog_lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._watchdog_stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="VRAMCoordinatorWatchdog",
                daemon=True,
            )
            self._watchdog_thread.start()
            logger.info("vram_watchdog_started")

    def stop_watchdog(self) -> None:
        """Stop the watchdog and wait for the thread to finish.

        The reference is cleared only once the thread has exited, so a
        timed-out join cannot be mistaken for a clean stop.
        """
        with self._watchdog_lock:
            self._watchdog_stop_event.set()
            thread = self._watchdog_thread
            if thread is None:
                return
            thread.join(timeout=WATCHDOG_INTERVAL_S * 2 + SYSFS_READ_TIMEOUT_S)
            if thread.is_alive():
                logger.warning("vram_watchdog_stop_timed_out")
                return
            self._watchdog_thread = None
            logger.info("vram_watchdog_stopped")

    def shutdown(self) -> None:
        """Full shutdown: watchdog + embedding service + accounting.

        Terminal: the coordinator holds no allocation afterwards, so a later
        ``acquire`` loads a fresh model instead of reporting a shared one.
        """
        self.stop_watchdog()
        with self._lock:
            self._allocations.clear()
            self._reservations.clear()
        self._embedding_service.shutdown()

    def track_operation_start(self, op_id: str) -> None:
        """Mark the start of a GPU operation for deadlock detection."""
        with self._lock:
            self._inflight[op_id] = time.monotonic()

    def track_operation_end(self, op_id: str) -> None:
        """Mark the end of a GPU operation."""
        with self._lock:
            self._inflight.pop(op_id, None)

    # -- Internals ----------------------------------------------------------

    def _detect_total_vram(self) -> None:
        """Resolve the amdgpu device and read its static VRAM sizes.

        Every attribute is read from the same device directory, so totals and
        usage cannot be mixed across devices on a multi-GPU host. Both reads
        are static and take no driver lock.
        """
        self._device_dir = self._resolve_device_dir()
        total_mb = self._read_attr_mb(_ATTR_VRAM_TOTAL)
        if total_mb is None:
            # No amdgpu attributes available: assume a common configuration
            # rather than failing, and remember that nothing was measured.
            self._total_vram_mb = 8192
            self._vram_detected_from_sysfs = False
            logger.warning(
                "vram_total_fallback_assumed",
                extra={"assumed_mb": 8192},
            )
            return

        self._total_vram_mb = total_mb
        self._vram_detected_from_sysfs = True
        logger.info(
            "vram_total_detected",
            extra={"total_mb": total_mb, "device": str(self._device_dir)},
        )

        visible_mb = self._read_attr_mb(_ATTR_VIS_VRAM_TOTAL)
        self._visible_vram_mb = visible_mb
        self._visible_window_trustworthy = (
            visible_mb is not None and visible_mb >= total_mb * VISIBLE_WINDOW_MIN_RATIO
        )
        if self._visible_window_trustworthy:
            logger.info(
                "vram_visible_window_covers_device",
                extra={"visible_mb": visible_mb, "total_mb": total_mb},
            )
        else:
            logger.warning(
                "vram_visible_window_smaller_than_device",
                extra={
                    "visible_mb": visible_mb,
                    "total_mb": total_mb,
                    "hint": (
                        "mem_info_vis_vram_used only counts VRAM inside the visible "
                        "window, so it cannot back a real budget; acquisitions will "
                        "fall back to CPU unless allow_contended_read=True"
                    ),
                },
            )

    def _resolve_device_dir(self) -> Path | None:
        """Device directory of the first DRM device that exposes amdgpu attributes."""
        for path_str in sorted(glob.glob(_DRM_DEVICE_GLOB)):
            device_dir = Path(path_str)
            if (device_dir / _ATTR_VRAM_TOTAL).exists():
                return device_dir
        return None

    def _read_attr_mb(self, attribute: str) -> int | None:
        """Read one VRAM attribute of the resolved device, in MB."""
        if self._device_dir is None:
            return None
        raw_bytes = _read_sysfs_int(
            str(self._device_dir / attribute),
            SYSFS_READ_TIMEOUT_S,
        )
        if raw_bytes is None:
            return None
        return int(raw_bytes / (1024.0 * 1024.0))

    def _get_free_vram_mb(self) -> int | None:
        """Free VRAM available for new reservations, or None if unknown.

        Convenience composition of :meth:`_measure_used_mb` (I/O) and
        :meth:`_free_mb_from_used` (accounting), used by the watchdog and
        available to callers that want a snapshot. ``acquire`` performs the two
        halves itself so the accounting is computed under the same lock that
        grants the reservation.
        """
        used_mb = self._measure_used_mb()
        with self._lock:
            return self._free_mb_from_used(used_mb)

    def _measure_used_mb(self) -> int | None:
        """Read VRAM usage from whichever source can be trusted.

        Performs I/O only, so it must be called without holding the lock.

        Sources, in order of preference:
            1. ``mem_info_vis_vram_used`` — lock-free, but only valid when the
               visible window covers the whole device (see module docstring).
            2. ``mem_info_vram_used`` — authoritative, but it takes the TTM
               ``lru_lock`` spinlock that every VRAM allocation also takes.
               Only when the operator enabled ``allow_contended_read=True``,
               and rate-limited.
            3. None, which makes the caller fall back to CPU.
        """
        if self._visible_window_trustworthy:
            return self._get_used_vram_lockfree()
        if not self._allow_contended_read:
            return None
        return self._get_used_vram_contended()

    def _free_mb_from_used(self, used_mb: int | None) -> int | None:
        """Budget left for new reservations, given a usage reading.

        Must be called with the lock held: the reservations already granted are
        part of the result, and the caller grants against this very figure. The
        budget is cumulative, so concurrent loads cannot each validate against
        the same free VRAM and over-commit the device.
        """
        if used_mb is None or self._total_vram_mb is None:
            return None
        free_mb = (
            self._total_vram_mb
            - used_mb
            - SYSTEM_VRAM_RESERVE_MB
            - SAFETY_HEADROOM_MB
            - self._reserved_total_mb()
        )
        return max(0, free_mb)

    def _get_used_vram_contended(self) -> int | None:
        """Read real VRAM usage from the contended path, rate-limited.

        This reads the authoritative counter, which is guarded by the TTM
        ``lru_lock`` spinlock. The interval slot and the in-flight flag are
        claimed before the read, so concurrent callers cannot turn the rate
        limit into a burst of readers: at most one read happens per interval. A
        caller that arrives while the first read is still in flight is served
        the cached value, or no measurement at all when nothing has been cached
        yet, in which case it falls back to CPU.
        """
        with self._contended_read_lock:
            if self._contended_read_inflight:
                return self._contended_used_mb
            cached = self._contended_used_mb
            if (
                cached is not None
                and (time.monotonic() - self._last_contended_read)
                < self._contended_read_min_interval_s
            ):
                return cached
            self._contended_read_inflight = True
            self._last_contended_read = time.monotonic()

        used_mb: int | None = None
        try:
            used_mb = self._read_attr_mb(_ATTR_VRAM_USED)
        finally:
            with self._contended_read_lock:
                if used_mb is not None:
                    self._contended_used_mb = used_mb
                self._contended_read_inflight = False
        return used_mb

    def _reserved_total_mb(self) -> int:
        """Total VRAM this coordinator has already granted."""
        with self._lock:
            return sum(alloc.reserved_mb for alloc in self._allocations.values())

    def _measurement_source(self) -> str:
        """Where the last successful measurement came from."""
        if self._visible_window_trustworthy:
            return "lockfree_visible_window"
        if self._allow_contended_read and self._contended_used_mb is not None:
            return "contended_mem_info_vram_used"
        return "unavailable"

    def _measurement_unavailable_reason(self) -> str:
        """Explain why VRAM usage could not be measured."""
        if not self._vram_detected_from_sysfs:
            return "vram_measurement_unavailable: no amdgpu sysfs on this host"
        if not self._visible_window_trustworthy:
            reason = (
                "vram_measurement_unavailable: visible_vram_window_too_small: "
                f"{self._visible_vram_mb}MB visible window on a "
                f"{self._total_vram_mb}MB device under-reports real usage"
            )
            if self._allow_contended_read:
                return f"{reason}; mem_info_vram_used is enabled but could not be read"
            return f"{reason}; pass allow_contended_read=True to read mem_info_vram_used"
        return "vram_measurement_unavailable"

    def _estimate_vram_mb(self, spec: ModelSpec, gpu_layers_actual: int) -> int:
        """Estimate the VRAM a fully or partially offloaded model needs.

        Approximate formula:
            vram = (model_size_mb * fraction_on_gpu) + (n_ctx * 0.5MB_overhead)

        The fraction comes from the layer counts alone — an explicit layer
        count is never reinterpreted as a percentage, and the estimate never
        exceeds the whole model:
            - ``-1`` offloads everything, so the full estimate applies.
            - ``k`` layers offload ``k / total_layers`` when the caller
              declared ``total_layers``.
            - Without ``total_layers`` the whole model is budgeted, since an
              under-estimate is the unsafe error.
        """
        if gpu_layers_actual < 0:
            # All layers: use the full estimate
            base_mb = spec.estimated_mb
        elif spec.total_layers is not None:
            layers = min(gpu_layers_actual, spec.total_layers)
            base_mb = int(spec.estimated_mb * layers / spec.total_layers)
        else:
            # Unknown layer count: budgeting less would under-reserve
            base_mb = spec.estimated_mb

        # Context overhead (KV cache): ~0.5 MB per 1k of context length
        ctx_overhead_mb = spec.n_ctx // 2000
        return base_mb + ctx_overhead_mb

    def _evict_idle(self, needed_mb: int) -> int:
        """Evict idle models until ``needed_mb`` is freed, then close them.

        Must be called **without** holding the coordinator lock: the evicted
        instances are closed through the embedding service, and the service
        takes the coordinator lock itself. Taking the two locks in this order
        would deadlock against ``EmbeddingService.get_or_create``.

        Returns the MB freed.
        """
        freed_mb, discarded = self._collect_evictions(needed_mb)
        for model_id in discarded:
            self._embedding_service._discard(model_id)
        return freed_mb

    def _collect_evictions(self, needed_mb: int) -> tuple[int, list[str]]:
        """Drop idle allocations from the table, most idle first.

        Only allocations with no live reference that have been idle for at
        least ``IDLE_EVICTION_TTL_S`` are eligible. Takes the coordinator lock,
        because the watchdog calls it too and mutating ``_allocations`` without
        the lock corrupts concurrent iteration.

        Returns ``(freed_mb, dropped_model_ids)``. Closing the dropped models is
        the caller's responsibility, and must happen outside the lock.
        """
        now = time.monotonic()
        discarded: list[str] = []
        freed_mb = 0
        with self._lock:
            candidates = [
                (now - alloc.last_used, model_id)
                for model_id, alloc in self._allocations.items()
                if alloc.ref_count == 0 and (now - alloc.last_used) >= IDLE_EVICTION_TTL_S
            ]
            candidates.sort(reverse=True)  # most idle first

            for _, model_id in candidates:
                allocation = self._allocations[model_id]
                logger.info(
                    "vram_evicting_idle_model",
                    extra={
                        "model_id": model_id,
                        "idle_seconds": now - allocation.last_used,
                        "reserved_mb": allocation.reserved_mb,
                    },
                )
                self._evictions += 1
                del self._allocations[model_id]
                self._drop_reservations_for(model_id)
                discarded.append(model_id)
                freed_mb += allocation.reserved_mb
                if freed_mb >= needed_mb:
                    break
        return freed_mb, discarded

    def _new_handle(self, model_id: str) -> str:
        """A reservation handle that is unique for this coordinator.

        Every grant, including every additional holder of a shared model, gets
        its own handle, so ``release`` can tell a valid handle from a stale or
        already-released one instead of acting twice on the same reservation.
        """
        return f"{model_id}::{next(self._handle_counter)}"

    def _register_shared_grant(self, model_id: str) -> tuple[str, int] | None:
        """Register one more holder of a model that is already loaded.

        Takes the coordinator lock itself, so it is safe to call from either
        side: ``acquire`` already holds it (it is reentrant) and the embedding
        service does not. The check, the increment and the handle are therefore
        atomic with respect to an eviction, which is what makes a grant imply a
        live allocation.

        Returns ``(handle, gpu_layers_actual)`` for the new holder, or ``None``
        when the model is no longer allocated: a caller that receives ``None``
        holds a stale instance and must acquire again instead of using it.
        """
        with self._lock:
            allocation = self._allocations.get(model_id)
            if allocation is None or allocation.ref_count <= 0:
                return None
            allocation.ref_count += 1
            allocation.last_used = time.monotonic()
            handle = self._new_handle(f"shared::{model_id}")
            self._reservations[handle] = model_id
            return handle, allocation.gpu_layers_actual

    def _is_allocated(self, model_id: str) -> bool:
        """Whether the coordinator currently holds a reservation for a model."""
        with self._lock:
            return model_id in self._allocations

    def _revoke_one_handle(self, model_id: str) -> None:
        """Revoke one reservation handle of a model.

        Called when one reference goes away without releasing the whole
        allocation, so the number of handles keeps matching the number of live
        references and a stale handle cannot decrement a later holder.
        """
        with self._lock:
            for reservation_id, owner in self._reservations.items():
                if owner == model_id:
                    del self._reservations[reservation_id]
                    return

    def _release_service_reference(self, model_id: str) -> bool:
        """Drop one reference held by the embedding service.

        Performed as a single operation under the coordinator lock, so the
        decision to release the allocation cannot be separated from the state
        it was based on. Returns True when the whole allocation was released,
        which means the caller must discard its cached instance.
        """
        with self._lock:
            allocation = self._allocations.get(model_id)
            if allocation is None or allocation.ref_count == 0:
                return False
            allocation.ref_count -= 1
            allocation.last_used = time.monotonic()
            if allocation.ref_count > 0:
                self._revoke_one_handle(model_id)
                logger.info(
                    "vram_model_no_longer_referenced",
                    extra={"model_id": model_id, "ref_count": allocation.ref_count},
                )
                return False
            self._allocations.pop(model_id, None)
            self._drop_reservations_for(model_id)
            logger.info(
                "vram_allocation_released",
                extra={
                    "model_id": model_id,
                    "gpu_layers_actual": allocation.gpu_layers_actual,
                    "reserved_mb": allocation.reserved_mb,
                },
            )
            return True

    def _decrement_ref(self, model_id: str) -> None:
        """Decrement the ref count. When it reaches 0, eligible for eviction."""
        with self._lock:
            alloc = self._allocations.get(model_id)
            if alloc is not None and alloc.ref_count > 0:
                alloc.ref_count -= 1
                alloc.last_used = time.monotonic()
                if alloc.ref_count == 0:
                    logger.info(
                        "vram_model_no_longer_referenced",
                        extra={"model_id": model_id},
                    )

    def _drop_reservations_for(self, model_id: str) -> None:
        """Forget every handle that pointed at a model holding no budget."""
        self._reservations = {
            reservation_id: owner
            for reservation_id, owner in self._reservations.items()
            if owner != model_id
        }

    def _abandon_allocation(self, model_id: str, reservation_id: str | None) -> None:
        """Give back the reservation held by a load that produced no model.

        Only the failed load's own reservation and handle are returned: if
        another caller holds a reference to the same model (through the flat
        API), the allocation stays, keeps consuming budget, and that caller's
        handle remains valid. A failed load holds no VRAM, so unlike a normal
        release this does not wait for the eviction TTL.
        """
        with self._lock:
            if reservation_id is not None:
                self._reservations.pop(reservation_id, None)
            allocation = self._allocations.get(model_id)
            if allocation is None:
                return
            if allocation.ref_count > 1:
                allocation.ref_count -= 1
                allocation.last_used = time.monotonic()
                return
            self._allocations.pop(model_id, None)
            self._drop_reservations_for(model_id)
            logger.info(
                "vram_allocation_abandoned",
                extra={"model_id": model_id, "reserved_mb": allocation.reserved_mb},
            )

    def _get_used_vram_lockfree(self) -> int | None:
        """Read *visible* VRAM usage without taking any driver lock."""
        return self._read_attr_mb(_ATTR_VIS_VRAM_USED)

    def _watchdog_loop(self) -> None:
        """Watchdog loop: detects deadlocks and runs health checks."""
        while not self._watchdog_stop_event.is_set():
            try:
                self._run_watchdog_iteration()
            except Exception as watch_exc:  # noqa: BLE001 - the loop must survive
                # A monitoring thread that dies on its first unexpected error
                # is worse than useless, so the failure is logged and the
                # next tick proceeds.
                logger.error(
                    "vram_watchdog_iteration_failed",
                    extra={"error": str(watch_exc)},
                )
            self._watchdog_stop_event.wait(timeout=WATCHDOG_INTERVAL_S)

    def _run_watchdog_iteration(self) -> None:
        """One watchdog iteration: health check + deadlock detection."""
        # 1. Health check: does the GPU respond?
        if not self._health_check():
            self._consecutive_health_failures += 1
            if self._consecutive_health_failures >= HEALTH_CHECK_FAILURE_THRESHOLD:
                self._handle_unhealthy_gpu()
        else:
            self._consecutive_health_failures = 0

        # 2. Deadlock detection: any operations running longer than the timeout?
        now = time.monotonic()
        with self._lock:
            stuck_ops = [
                (op_id, start)
                for op_id, start in self._inflight.items()
                if (now - start) > DEADLOCK_DETECTION_TIMEOUT_S
            ]
        if stuck_ops:
            logger.critical(
                "vram_watchdog_detected_potential_deadlock",
                extra={
                    "stuck_operations": [
                        {"op_id": op, "duration_s": now - start} for op, start in stuck_ops
                    ],
                    "consecutive_health_failures": self._consecutive_health_failures,
                },
            )
            # An operation that never completed is treated as lost and stops
            # being tracked: otherwise every following tick would detect the
            # same entry and trigger recovery again, closing models that were
            # loaded in the meantime.
            with self._lock:
                for op_id, _start in stuck_ops:
                    self._inflight.pop(op_id, None)
            self._handle_unhealthy_gpu()

        # 3. Periodic eviction if the budget is tight (free below 10% of the
        # device). Skipped when usage cannot be measured: "unknown" is not
        # "exhausted".
        free_mb = self._get_free_vram_mb()
        if free_mb is not None and self._total_vram_mb:
            threshold_mb = self._total_vram_mb * (1 - EVICTION_THRESHOLD_PCT)
            if free_mb < threshold_mb:
                self._evict_idle(int(threshold_mb - free_mb))

    def _health_check(self) -> bool:
        """Verify that the GPU responds.

        A successful lock-free read means the driver is serving requests.
        When the read fails, the meaning depends on the host: on a machine
        with amdgpu sysfs it is a real failure (the GPU stopped answering),
        while on a host without amdgpu sysfs it is simply UNKNOWN and must
        not be treated as a fault.
        """
        if self._get_used_vram_lockfree() is not None:
            self._health = GPUHealth.HEALTHY
            return True
        if not self._vram_detected_from_sysfs:
            # No amdgpu here: unable to tell, and that is not a failure.
            self._health = GPUHealth.UNKNOWN
            return True
        if self._health is not GPUHealth.UNHEALTHY:
            self._health = GPUHealth.DEGRADED
        return False

    def _handle_unhealthy_gpu(self) -> None:
        """Recovery for an unhealthy GPU.

        Strategy:
            1. Mark the GPU as UNHEALTHY, which blocks new reservations.
            2. Drop every allocation and close the models, so the memory is
               returned to the driver instead of leaking. Callers that still
               hold a reference to one of those models must not use it again.
            3. Wait DEADLOCK_DETECTION_TIMEOUT_S for the driver to recover.
            4. Re-check health. The GPU returns to HEALTHY as soon as a
               lock-free read succeeds again; otherwise it stays UNHEALTHY and
               callers keep falling back to CPU.
        """
        logger.critical(
            "vram_watchdog_activating_recovery",
            extra={
                "consecutive_health_failures": self._consecutive_health_failures,
                "current_allocations": len(self._allocations),
            },
        )
        with self._lock:
            self._health = GPUHealth.UNHEALTHY
            released = list(self._allocations.keys())
            for model_id in released:
                alloc = self._allocations[model_id]
                if alloc.ref_count > 0:
                    logger.warning(
                        "vram_force_release_in_use_model",
                        extra={
                            "model_id": model_id,
                            "ref_count": alloc.ref_count,
                        },
                    )
            self._allocations.clear()
            self._reservations.clear()

        # Close the instances outside the lock (and outside the service lock)
        for model_id in released:
            self._embedding_service._discard(model_id)

        # Wait and re-check
        time.sleep(DEADLOCK_DETECTION_TIMEOUT_S)
        if self._health_check():
            self._health = GPUHealth.HEALTHY
            self._consecutive_health_failures = 0
            logger.info("vram_watchdog_recovery_successful")
        else:
            logger.error(
                "vram_watchdog_recovery_failed_manual_intervention_required",
            )


# --- Singleton accessor ---

_coordinator: VRAMCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_coordinator() -> VRAMCoordinator:
    """Get the coordinator singleton instance.

    Thread-safe. Starts the watchdog automatically on first use.
    """
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = VRAMCoordinator()
            _coordinator.start_watchdog()
        return _coordinator


def reset_coordinator() -> None:
    """Reset the singleton (for tests). Do not use in production."""
    global _coordinator
    with _coordinator_lock:
        coordinator = _coordinator
        _coordinator = None
    # Shut down outside the module lock: stopping the watchdog joins a thread
    # and must not block get_coordinator() for everyone else.
    if coordinator is not None:
        coordinator.shutdown()


def shutdown_coordinator() -> None:
    """Full shutdown of the singleton (on process exit)."""
    global _coordinator
    with _coordinator_lock:
        coordinator = _coordinator
        _coordinator = None
    if coordinator is not None:
        coordinator.shutdown()
