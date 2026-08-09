from mlx_vlm_gateway.crash_diagnostics import (
    capture_log_position,
    fresh_log_has_out_of_memory,
    latest_memory_was_critical,
)
from mlx_vlm_gateway.memory_monitor import MemorySample


def test_fresh_log_check_ignores_older_out_of_memory_message(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("[METAL] Command buffer execution failed: Insufficient Memory\n")
    position = capture_log_position(str(log))

    with log.open("a") as stream:
        stream.write("worker connection reset\n")

    assert not fresh_log_has_out_of_memory(str(log), position)


def test_fresh_log_check_finds_new_out_of_memory_message(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("worker ready\n")
    position = capture_log_position(str(log))

    with log.open("a") as stream:
        stream.write(
            "[METAL] Command buffer execution failed: Insufficient Memory\n"
        )

    assert fresh_log_has_out_of_memory(str(log), position)


def _memory_sample(timestamp: float, pressure: str, pid: int = 123) -> MemorySample:
    return MemorySample(
        timestamp=timestamp,
        pressure=pressure,
        available_bytes=100,
        worker_bytes=50,
        worker_pid=pid,
    )


def test_latest_memory_sample_must_be_fresh_and_critical():
    assert latest_memory_was_critical(
        [_memory_sample(99.0, "critical")], 123, now=100.0
    )
    assert not latest_memory_was_critical(
        [_memory_sample(90.0, "critical")], 123, now=100.0
    )
    assert not latest_memory_was_critical(
        [
            _memory_sample(99.0, "critical"),
            _memory_sample(100.0, "normal"),
        ],
        123,
        now=100.0,
    )
