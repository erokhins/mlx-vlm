"""Idle auto-unload watchdog.

When the ``auto_unload_time`` setting is set (seconds, via
``MLX_VLM_AUTO_UNLOAD_TIME``), a background thread unloads the model after
that much inference inactivity to free memory. The server stays in phase
"ready"; the next inference request reloads the model lazily.
"""

import logging
import os
import time
from threading import Lock, Thread

from ..runtime import runtime
from .lifecycle import PHASE_READY, lifecycle
from .state import metrics_in_flight, reload_lock, serving_config

logger = logging.getLogger("mlx_vlm.server")

AUTO_UNLOAD_TIME_ENV = "MLX_VLM_AUTO_UNLOAD_TIME"
AUTO_UNLOAD_POLL_S = 10

_state_lock = Lock()
_started = False


def auto_unload_seconds():
    raw = os.environ.get(AUTO_UNLOAD_TIME_ENV)
    try:
        value = int(raw) if raw else 0
    except ValueError:
        value = 0
    return value if value > 0 else None


def _idle_seconds() -> float:
    """Seconds since the last inference activity (or the model load)."""
    last_request_at = 0.0
    if runtime.metrics is not None:
        summary = runtime.metrics.snapshot()["summary"]
        last_request_at = float(summary.get("last_request_at") or 0.0)
    loaded_at = float(serving_config.get("loaded_at") or 0.0)
    anchor = max(last_request_at, loaded_at)
    if anchor <= 0:
        return 0.0
    return max(0.0, time.time() - anchor)


def ensure_idle_watchdog(deps) -> None:
    global _started
    with _state_lock:
        if _started:
            return
        _started = True
    Thread(
        target=_idle_watchdog_loop,
        args=(deps,),
        daemon=True,
        name="auto-unload-watchdog",
    ).start()


def _idle_watchdog_loop(deps) -> None:
    while True:
        time.sleep(AUTO_UNLOAD_POLL_S)
        try:
            _maybe_auto_unload(deps)
        except Exception:
            logger.exception("Auto-unload watchdog error")


def _maybe_auto_unload(deps) -> None:
    timeout = auto_unload_seconds()
    if timeout is None:
        return
    if lifecycle.phase() != PHASE_READY:
        return
    if not deps.model_cache_registry().for_kind("text_generation"):
        return
    if not reload_lock.acquire(blocking=False):
        return
    try:
        # Re-check under the lock so we never unload during a settings
        # change or while a request is running.
        if lifecycle.phase() != PHASE_READY or metrics_in_flight() > 0:
            return
        idle_s = _idle_seconds()
        if idle_s < timeout:
            return
        logger.info(
            "Auto-unload: no inference activity for %.0fs (limit %ds); "
            "unloading model. The next request triggers a guarded reload.",
            idle_s,
            timeout,
        )
        deps.unload_model_sync()
        lifecycle.set_phase(PHASE_READY, "model auto-unloaded after idle timeout")
    finally:
        reload_lock.release()
