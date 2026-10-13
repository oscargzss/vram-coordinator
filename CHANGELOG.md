# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-12

First release.

### Added

- `VRAMCoordinator`: a cumulative VRAM budget with `acquire()`, `release()` and a
  `reserve()` context manager. Every granted reservation is subtracted from the
  next computation, under the same lock that grants it, so concurrent callers
  cannot over-commit the device.
- Measurement strategy: the lock-free `mem_info_vis_vram_used` reading is used
  only when the visible window covers the device. Otherwise the coordinator
  reports that it cannot measure and the caller falls back to CPU. The
  authoritative `mem_info_vram_used` counter is available through
  `allow_contended_read=True`, read at most once per
  `CONTENDED_READ_MIN_INTERVAL_S`.
- `EmbeddingService`: one model instance per `model_id`, shared through reference
  counting, closed when the last holder releases it.
- Idle eviction after `IDLE_EVICTION_TTL_S`, and under pressure below 10 % of the
  device, closing the models it drops so their memory returns to the driver.
- Watchdog thread: health checks, detection of operations that never complete,
  and recovery that drops the reservations it cannot account for.
- `ModelSpec` with explicit layer semantics: `n_gpu_layers=-1` requests all layers
  and is denied when the model does not fit, and an explicit count is passed to
  the loader unchanged. `total_layers` sizes partial offloads.
- Sysfs reads bounded by `SYSFS_READ_TIMEOUT_S`, shared per path so a device that
  stops answering cannot accumulate reader threads.
- 93 tests, no runtime dependencies, Python 3.10 to 3.14.
