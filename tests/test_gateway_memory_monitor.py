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
