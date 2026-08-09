from mlx_vlm_gateway.crash_diagnostics import (
    capture_log_position,
    fresh_log_has_out_of_memory,
)


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
