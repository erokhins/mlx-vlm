import mlx_vlm_gateway.memory_monitor as memory_monitor
from mlx_vlm_gateway.memory_monitor import MemorySample, RecentMemorySamples


def _sample(number: int) -> MemorySample:
    return MemorySample(
        timestamp=float(number),
        pressure="normal",
        available_bytes=number,
        worker_bytes=number,
        worker_pid=123,
    )


def test_recent_memory_samples_keep_only_the_configured_limit():
    history = RecentMemorySamples(limit=20)

    for number in range(25):
        history.append(_sample(number))

    samples = history.snapshot()
    assert len(samples) == 20
    assert samples[0].timestamp == 5.0
    assert samples[-1].timestamp == 24.0


def test_memory_sample_uses_standard_macos_commands(monkeypatch):
    responses = {
        ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1",
        ("sysctl", "-n", "hw.pagesize"): "4096",
        ("sysctl", "-n", "hw.memsize"): str(100 * 4096),
        ("vm_stat",): """
Pages active: 40.
Pages wired down: 10.
Pages occupied by compressor: 5.
""",
        ("ps", "-o", "rss=", "-p", "123"): "2048",
    }
    monkeypatch.setattr(
        memory_monitor,
        "_run_command",
        lambda *command: responses.get(command),
    )
    monkeypatch.setattr(memory_monitor.time, "time", lambda: 42.0)

    sample = memory_monitor.read_memory_sample(123)

    assert sample == MemorySample(
        timestamp=42.0,
        pressure="normal",
        available_bytes=45 * 4096,
        worker_bytes=2048 * 1024,
        worker_pid=123,
    )


def test_memory_sample_tolerates_unavailable_macos_commands(monkeypatch):
    monkeypatch.setattr(memory_monitor, "_run_command", lambda *command: None)

    sample = memory_monitor.read_memory_sample(123)

    assert sample.pressure == "unknown"
    assert sample.available_bytes is None
    assert sample.worker_bytes is None
