import pytest

from mlx_vlm_shared.errors import is_out_of_memory_error


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        RuntimeError("[malloc] Unable to allocate 4096 bytes"),
        RuntimeError(
            "[metal::malloc] Attempting to allocate 17179869184 bytes "
            "greater than the maximum allowed buffer size"
        ),
        RuntimeError(
            "[METAL] Command buffer execution failed: Insufficient Memory"
        ),
        RuntimeError("kIOGPUCommandBufferCallbackErrorOutOfMemory"),
    ],
)
def test_recognizes_confirmed_out_of_memory_errors(error):
    assert is_out_of_memory_error(error)


def test_recognizes_out_of_memory_in_exception_cause():
    try:
        try:
            raise MemoryError("allocation failed")
        except MemoryError as error:
            raise RuntimeError("generation failed") from error
    except RuntimeError as error:
        assert is_out_of_memory_error(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("GPU Timeout Error"),
        RuntimeError("There is no Stream(gpu, 2) in current thread"),
        RuntimeError("Command buffer submissions ignored"),
        RuntimeError("Connection reset by peer"),
        RuntimeError("Worker exited after SIGKILL (-9)"),
        RuntimeError("An unrelated RuntimeError"),
    ],
)
def test_does_not_guess_out_of_memory_from_unrelated_errors(error):
    assert not is_out_of_memory_error(error)
