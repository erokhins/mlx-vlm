"""Error classification shared by the gateway and inference worker."""

from collections.abc import Iterator


_MLX_OUT_OF_MEMORY_MARKERS = (
    "[malloc] unable to allocate",
    "[metal::malloc] attempting to allocate",
    "command buffer execution failed: insufficient memory",
    "kiogpucommandbuffercallbackerroroutofmemory",
)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit or implicit causes once each."""
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_out_of_memory_error(error: BaseException) -> bool:
    """Return whether an exception contains a confirmed MLX OOM signal."""
    for item in _exception_chain(error):
        if isinstance(item, MemoryError):
            return True

        message = str(item).lower()
        if any(marker in message for marker in _MLX_OUT_OF_MEMORY_MARKERS):
            return True

    return False
