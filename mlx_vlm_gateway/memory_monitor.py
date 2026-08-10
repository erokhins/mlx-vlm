"""Bounded, OS-level memory observations owned by the gateway."""

import subprocess
import time
from collections import deque
from dataclasses import dataclass


MEMORY_SAMPLE_INTERVAL_S = 10.0
MEMORY_SAMPLE_LIMIT = 20

_PRESSURE_NORMAL = 1
_PRESSURE_WARNING = 2
_PRESSURE_CRITICAL = 4

_VM_STAT_FIELDS = {
    "Pages active": "active",
    "Pages wired down": "wired",
    "Pages occupied by compressor": "compressed",
}


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


def _run_command(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sysctl_int(name: str) -> int | None:
    output = _run_command("sysctl", "-n", name)
    if output is None:
        return None
    try:
        return int(output)
    except ValueError:
        return None


def _system_pressure() -> str:
    value = _sysctl_int("kern.memorystatus_vm_pressure_level")
    if value is None:
        return "unknown"
    if value & _PRESSURE_CRITICAL:
        return "critical"
    if value & _PRESSURE_WARNING:
        return "warning"
    if value & _PRESSURE_NORMAL:
        return "normal"
    return "unknown"


def _available_memory_bytes() -> int | None:
    page_size = _sysctl_int("hw.pagesize")
    total = _sysctl_int("hw.memsize")
    vm_stat = _run_command("vm_stat")
    if page_size is None or total is None or vm_stat is None:
        return None

    pages = {}
    for line in vm_stat.splitlines():
        name, separator, raw_value = line.partition(":")
        field = _VM_STAT_FIELDS.get(name)
        if not separator or field is None:
            continue
        try:
            pages[field] = int(raw_value.strip().rstrip("."))
        except ValueError:
            return None
    if pages.keys() != set(_VM_STAT_FIELDS.values()):
        return None

    used_pages = pages["active"] + pages["wired"] + pages["compressed"]
    return max(0, total - used_pages * page_size)


def _worker_memory_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    output = _run_command("ps", "-o", "rss=", "-p", str(pid))
    if output is None:
        return None
    try:
        return int(output) * 1024
    except ValueError:
        return None


def read_memory_sample(worker_pid: int | None) -> MemorySample:
    """Read memory state without calling the worker, MLX, or Metal."""
    return MemorySample(
        timestamp=time.time(),
        pressure=_system_pressure(),
        available_bytes=_available_memory_bytes(),
        worker_bytes=_worker_memory_bytes(worker_pid),
        worker_pid=worker_pid,
    )
