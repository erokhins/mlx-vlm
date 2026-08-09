"""Bounded, OS-level memory observations owned by the gateway."""

import ctypes
import time
from collections import deque
from dataclasses import dataclass


MEMORY_SAMPLE_INTERVAL_S = 1.0
MEMORY_SAMPLE_LIMIT = 20

_HOST_VM_INFO64 = 4
_HOST_VM_INFO64_COUNT = 38
_FREE_COUNT = 0
_INACTIVE_COUNT = 2
_SPECULATIVE_COUNT = 23

_RUSAGE_INFO_V4 = 4
_FOOTPRINT_SLOT = 9
_RUSAGE_BUFFER_SLOTS = 40

_PRESSURE_NORMAL = 1
_PRESSURE_WARNING = 2
_PRESSURE_CRITICAL = 4


try:
    _libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    _libsystem.mach_host_self.restype = ctypes.c_uint
    _libsystem.host_page_size.argtypes = (
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    )
    _libsystem.host_statistics64.argtypes = (
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint),
    )
    _libsystem.sysctlbyname.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    _libproc.proc_pid_rusage.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    )
except (AttributeError, OSError):  # non-macOS development and tests
    _libsystem = None
    _libproc = None


@dataclass(frozen=True)
class MemorySample:
    timestamp: float
    pressure: str
    available_bytes: int | None
    worker_bytes: int | None
    worker_pid: int | None


class RecentMemorySamples:
    """Keep only the most recent fixed number of observations in RAM."""

    def __init__(self, limit: int = MEMORY_SAMPLE_LIMIT):
        self._samples: deque[MemorySample] = deque(maxlen=limit)

    def append(self, sample: MemorySample) -> None:
        self._samples.append(sample)

    def snapshot(self) -> tuple[MemorySample, ...]:
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


def _system_pressure() -> str:
    if _libsystem is None:
        return "unknown"
    try:
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        result = _libsystem.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        if result != 0:
            return "unknown"
        if value.value & _PRESSURE_CRITICAL:
            return "critical"
        if value.value & _PRESSURE_WARNING:
            return "warning"
        if value.value & _PRESSURE_NORMAL:
            return "normal"
    except Exception:
        pass
    return "unknown"


def _available_memory_bytes() -> int | None:
    if _libsystem is None:
        return None
    try:
        host = _libsystem.mach_host_self()
        page_size = ctypes.c_uint(0)
        if _libsystem.host_page_size(host, ctypes.byref(page_size)) != 0:
            return None
        info = (ctypes.c_uint32 * _HOST_VM_INFO64_COUNT)()
        count = ctypes.c_uint(_HOST_VM_INFO64_COUNT)
        if (
            _libsystem.host_statistics64(
                host, _HOST_VM_INFO64, info, ctypes.byref(count)
            )
            != 0
        ):
            return None
        available_pages = (
            info[_FREE_COUNT]
            + info[_INACTIVE_COUNT]
            + info[_SPECULATIVE_COUNT]
        )
        return int(available_pages * page_size.value)
    except Exception:
        return None


def _worker_footprint_bytes(pid: int | None) -> int | None:
    if _libproc is None or pid is None:
        return None
    try:
        buffer = (ctypes.c_uint64 * _RUSAGE_BUFFER_SLOTS)()
        if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V4, buffer) != 0:
            return None
        return int(buffer[_FOOTPRINT_SLOT])
    except Exception:
        return None


def read_memory_sample(worker_pid: int | None) -> MemorySample:
    """Read memory state without calling the worker, MLX, or Metal."""
    return MemorySample(
        timestamp=time.time(),
        pressure=_system_pressure(),
        available_bytes=_available_memory_bytes(),
        worker_bytes=_worker_footprint_bytes(worker_pid),
        worker_pid=worker_pid,
    )
