"""Small, bounded checks for evidence left by a failed worker."""

import os

from mlx_vlm_shared.errors import is_out_of_memory_message


MAX_FRESH_LOG_BYTES = 64 * 1024


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
