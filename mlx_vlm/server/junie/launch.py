"""Launch the server from the persistent config file.

``python -m mlx_vlm.server.junie`` reads ``JUNIE_SERVER_CONFIG`` and turns
the launch-time settings (host/port, prefill tuning, seed request, ...)
into the equivalent ``mlx_vlm.server`` command line — so start.sh stays a
dumb bootstrapper and every serving setting lives in one file. Runtime
settings (model, KV quantization, ...) keep flowing through the lifespan
config read as before.

Extra command-line arguments are appended after the config-derived ones
(argparse last-wins), so ad-hoc overrides still work:

    python -m mlx_vlm.server.junie --log-level DEBUG
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_CONFIG, config_path, load_config


def _default_seed_request() -> Optional[str]:
    """The repo's bundled Junie seed prompt, when running from a checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "research" / "junie.json"
    return str(path) if path.is_file() else None


def apply_inference_env(cfg: dict) -> None:
    """Export the inference env vars (APC, ngram cap) from the config.

    These are read by the runtime at model-load / draft time, not parsed
    as server flags, so the launcher sets them before the server starts.
    """

    def set_or_unset(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    set_or_unset("APC_ENABLED", "1" if cfg.get("apc_enabled") else "0")
    set_or_unset("APC_EXACT_SESSIONS", cfg.get("apc_exact_sessions"))
    set_or_unset("APC_SESSION_CHECKPOINTS", cfg.get("apc_session_checkpoints"))
    disk_path = cfg.get("apc_disk_path")
    if disk_path is None:
        base = config_path()
        disk_path = (
            os.path.join(os.path.dirname(base), "apc-cache") if base else None
        )
    set_or_unset(
        "APC_DISK_PATH", os.path.expanduser(disk_path) if disk_path else None
    )
    set_or_unset("MLX_VLM_NGRAM_MAX", cfg.get("ngram_max"))
    set_or_unset(
        "MLX_VLM_MAX_CONCURRENT_REQUESTS", cfg.get("max_concurrent_requests")
    )


def build_argv(cfg: dict) -> List[str]:
    argv = [
        "--host",
        str(cfg.get("host") or DEFAULT_CONFIG["host"]),
        "--port",
        str(cfg.get("port") or DEFAULT_CONFIG["port"]),
        "--prefill-step-size",
        str(cfg.get("prefill_step_size") or DEFAULT_CONFIG["prefill_step_size"]),
    ]
    if cfg.get("int8_prefill"):
        argv.append("--int8-prefill")
    if cfg.get("preserve_thinking"):
        argv.append("--preserve-thinking")
    if cfg.get("log_raw_tokens"):
        argv.append("--log-raw-tokens")
    seed = cfg.get("seed_request")
    if seed is None:
        seed = _default_seed_request()
    if seed:
        argv.extend(["--seed-request", str(seed)])
    return argv


def main() -> None:
    from ..cli import main as cli_main

    cfg = load_config()
    if cfg is None:
        # JUNIE_SERVER_CONFIG not set: fall back to stock flag behavior.
        cli_main()
        return
    apply_inference_env(cfg)
    sys.argv = [sys.argv[0], *build_argv(cfg), *sys.argv[1:]]
    cli_main()
