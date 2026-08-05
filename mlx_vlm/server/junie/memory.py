"""Memory stats for GET /status.

Three numbers the mac app cares about:
- total: the server process's physical footprint (what Activity Monitor
  reports — includes the Metal buffers, which plain RSS undercounts),
- peak: the lifetime maximum of that footprint,
- kv_cache: in-RAM KV held by the prefix cache (APC) — the one component
  that can be reduced without unloading the model (POST /v1/cache/reset,
  or lower APC_EXACT_SESSIONS).
"""

import ctypes
import os

import mlx.core as mx

from ..runtime import runtime

# proc_pid_rusage(pid, RUSAGE_INFO_V4, buf) fills struct rusage_info_v4:
# a 16-byte uuid followed by uint64 fields (<sys/resource.h>). Viewed as a
# uint64 array the uuid occupies slots 0-1, ri_phys_footprint is slot 9 and
# ri_lifetime_max_phys_footprint slot 30.
_RUSAGE_INFO_V4 = 4
_FOOTPRINT_SLOT = 9
_PEAK_FOOTPRINT_SLOT = 30
_BUF_SLOTS = 40

try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
except OSError:  # non-macOS
    _libproc = None


def _process_footprints():
    """(current, lifetime peak) physical footprint in bytes, or None."""
    if _libproc is None:
        return None
    buf = (ctypes.c_uint64 * _BUF_SLOTS)()
    if _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, buf) != 0:
        return None
    return int(buf[_FOOTPRINT_SLOT]), int(buf[_PEAK_FOOTPRINT_SLOT])


def kv_cache_bytes() -> int:
    """In-RAM KV bytes held by the APC (sessions + block pool)."""
    manager = runtime.apc_manager
    if manager is None:
        return 0
    try:
        snap = manager.stats_snapshot()
    except Exception:
        return 0
    total = int(snap.get("resident_bytes") or 0)
    for session in snap.get("exact_sessions") or []:
        total += int(session.get("kv_bytes") or 0)
    return total


def memory_stats() -> dict:
    footprints = _process_footprints()
    if footprints is not None:
        total, peak = footprints
    else:
        total = mx.get_active_memory() + mx.get_cache_memory()
        peak = mx.get_peak_memory()
    return {
        "total_gb": round(total / 2**30, 2),
        "peak_gb": round(peak / 2**30, 2),
        "kv_cache_gb": round(kv_cache_bytes() / 2**30, 2),
    }
