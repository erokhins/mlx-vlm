"""Mutable serving state shared by the junie control plane modules."""

from threading import Lock

from ..runtime import runtime

# Serializes model-serving reconfiguration: /apply_settings restarts and
# guarded reloads take this lock so they never overlap.
reload_lock = Lock()

# Model/adapter the server is configured to serve. Set by the startup loader
# and updated by /apply_settings, so a model restart knows what to reload
# even after MLX_VLM_PRELOAD_MODEL has been popped from the environment.
serving_config = {"model_path": None, "adapter_path": None, "loaded_at": None}


def metrics_in_flight() -> int:
    if runtime.metrics is None:
        return 0
    summary = runtime.metrics.snapshot()["summary"]
    return int(summary.get("in_flight", 0) or 0)
