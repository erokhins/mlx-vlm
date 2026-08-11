"""Small, bounded checks for evidence left by a failed worker."""

import os
import time
from collections.abc import Iterable

from mlx_vlm_shared.errors import is_out_of_memory_message

from .memory_monitor import MemorySample


MAX_FRESH_LOG_BYTES = 64 * 1024
RECENT_MEMORY_SAMPLE_MAX_AGE_S = 3.0


def capture_log_position(path: str | None) -> int | None:
    """Return the current worker-log size, or None when it is unavailable."""
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def fresh_log_has_out_of_memory(
    path: str | None, position: int | None
) -> bool:
    """Check only bounded log text written after ``position`` for MLX OOM."""
    if not path or position is None:
        return False
    try:
        size = os.path.getsize(path)
        if size < position:
            return False
        start = max(position, size - MAX_FRESH_LOG_BYTES)
        with open(path, "rb") as log:
            log.seek(start)
            fresh_text = log.read(MAX_FRESH_LOG_BYTES).decode(
                "utf-8", errors="replace"
            )
    except OSError:
        return False
    return is_out_of_memory_message(fresh_text)


def latest_memory_was_critical(
    samples: Iterable[MemorySample],
    worker_pid: int | None,
    *,
    now: float | None = None,
) -> bool:
    """Return whether this worker's latest sample is fresh and critical."""
    if worker_pid is None:
        return False
    current_time = time.time() if now is None else now
    for sample in reversed(tuple(samples)):
        if sample.worker_pid != worker_pid:
            continue
        age = current_time - sample.timestamp
        return (
            0 <= age <= RECENT_MEMORY_SAMPLE_MAX_AGE_S
            and sample.pressure == "critical"
        )
    return False
