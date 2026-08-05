"""Persistent server settings.

start.sh points ``JUNIE_SERVER_CONFIG`` at server-config.json inside the
junie-local data dir (next to the model weights and the APC disk cache).
The file is read once at startup — before the model preload — and
rewritten whenever /apply_settings succeeds, so settings survive restarts.

Only active when ``JUNIE_SERVER_CONFIG`` is set; a bare
``python -m mlx_vlm.server`` keeps the stock env/flag behavior.
"""

import json
import logging
import os
import tempfile
from typing import Optional

from .watchdog import AUTO_UNLOAD_TIME_ENV

logger = logging.getLogger("mlx_vlm.server")

CONFIG_PATH_ENV = "JUNIE_SERVER_CONFIG"

DEFAULT_KV_QUANT_BITS = 8

DEFAULT_CONFIG = {
    # --- Runtime settings (GET /current_settings, POST /apply_settings). ---
    # The same ids start.sh downloads and registers with Junie; keep in sync.
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "draft_kind": "mtp",
    "max_context_length": None,
    "kv_quantization": False,
    "auto_unload_time": None,
    # --- Launch settings (read once by `python -m mlx_vlm.server.junie`;
    # edit by hand while the server is stopped). ---
    "host": "0.0.0.0",
    "port": 8085,
    # W8A8 int8 prefill on the M5 neural accelerators (research/int8-nax).
    "int8_prefill": True,
    # Tokens per prefill chunk. Each chunk materializes per-layer attention
    # scores of step x context, so this directly scales peak memory on long
    # contexts (sweep on 31.7k prompts: 4096 -> 36.9 GB peak @980 tok/s,
    # 1024 -> 28.6 GB @910 tok/s; warm-workload speed identical).
    "prefill_step_size": 1024,
    # Required for APC warm hits to survive new user turns (the Qwen3.6
    # chat template otherwise re-renders older assistant turns).
    "preserve_thinking": True,
    # Chat request whose rendered prompt is a stable cross-session prefix,
    # prefilled and pinned at startup. null = use the repo's
    # research/junie.json when present; "" disables the warmup.
    "seed_request": None,
    "log_raw_tokens": True,
    # Automatic Prefix Caching. Qwen3.6 is a hybrid linear-attention model,
    # so APC uses session storage: ONE shared full-attention KV set per
    # conversation plus small recurrent-state checkpoints at the last few
    # prefix lengths (a warm hit resumes from any recent checkpoint,
    # including one before a mid-history edit).
    "apc_enabled": True,
    # Concurrent conversations kept warm / resumable positions each.
    "apc_exact_sessions": 2,
    "apc_session_checkpoints": 8,
    # Disk tier for the pinned seed snapshot (survives restarts). null =
    # "apc-cache" next to this config file; "" disables the disk tier.
    "apc_disk_path": None,
    # Cap for the n-gram prompt-lookup draft window (it doubles on full
    # accepts). Every drafted block is verified in one forward whose
    # per-layer GDN states scale with block length — window 32 transiently
    # pins ~11 GB during decode on 30k contexts; window 8 keeps the n-gram
    # speedup at ~1 GB (same wall time on the junie replay).
    "ngram_max": 8,
}


def _is_positive_int_or_none(value) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _is_int_in(low, high):
    def check(value):
        return isinstance(value, int) and not isinstance(value, bool) and (
            low <= value <= high
        )

    return check


_VALIDATORS = {
    "model_name": lambda v: v is None or (isinstance(v, str) and v.strip()),
    "draft_model": lambda v: v is None or (isinstance(v, str) and v.strip()),
    "draft_kind": lambda v: v is None or v in ("dflash", "eagle3", "mtp"),
    "max_context_length": _is_positive_int_or_none,
    "kv_quantization": lambda v: isinstance(v, bool),
    "auto_unload_time": _is_positive_int_or_none,
    "host": lambda v: isinstance(v, str) and v.strip(),
    "port": _is_int_in(1, 65535),
    "int8_prefill": lambda v: isinstance(v, bool),
    "prefill_step_size": _is_int_in(1, 1 << 20),
    "preserve_thinking": lambda v: isinstance(v, bool),
    "seed_request": lambda v: v is None or isinstance(v, str),
    "log_raw_tokens": lambda v: isinstance(v, bool),
    "apc_enabled": lambda v: isinstance(v, bool),
    "apc_exact_sessions": _is_int_in(0, 64),
    "apc_session_checkpoints": _is_int_in(1, 64),
    "apc_disk_path": lambda v: v is None or isinstance(v, str),
    "ngram_max": _is_int_in(1, 1024),
}


def config_path() -> Optional[str]:
    return os.environ.get(CONFIG_PATH_ENV) or None


def _sanitize(raw: dict) -> dict:
    """Merge a raw file dict over the defaults, dropping invalid values."""
    cfg = dict(DEFAULT_CONFIG)
    for key, value in raw.items():
        validator = _VALIDATORS.get(key)
        if validator is None:
            cfg[key] = value  # unknown keys pass through (forward compat)
        elif validator(value):
            cfg[key] = value
        else:
            logger.warning(
                "Config: ignoring invalid %r value %r (keeping %r)",
                key,
                value,
                cfg.get(key),
            )
    return cfg


def _write(path: str, cfg: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".server-config-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_config() -> Optional[dict]:
    """Read the config (creating it with defaults when missing).

    Returns None when JUNIE_SERVER_CONFIG is not set. A corrupt file is
    kept aside as <path>.invalid and replaced with the defaults.
    """
    path = config_path()
    if not path:
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("config root must be a JSON object")
    except FileNotFoundError:
        logger.info("Config: %s not found; creating it with defaults.", path)
        _write(path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    except (OSError, ValueError) as e:
        broken = path + ".invalid"
        logger.warning(
            "Config: cannot read %s (%s); recreating with defaults "
            "(old file kept at %s).",
            path,
            e,
            broken,
        )
        try:
            os.replace(path, broken)
        except OSError:
            pass
        _write(path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    cfg = _sanitize(raw)
    missing = set(cfg) - set(raw)
    if missing:
        # Upgrade the file in place so newly introduced fields are visible
        # for hand-editing.
        try:
            _write(path, cfg)
            logger.info(
                "Config: added default fields %s to %s", sorted(missing), path
            )
        except OSError:
            pass
    return cfg


def save_settings(updates: dict) -> None:
    """Merge applied settings into the config file (no-op when unset)."""
    path = config_path()
    if not path:
        return
    cfg = load_config() or dict(DEFAULT_CONFIG)
    cfg = _sanitize({**cfg, **updates})
    try:
        _write(path, cfg)
        logger.info("Config: saved %s to %s", sorted(updates), path)
    except OSError as e:
        logger.warning("Config: failed to save settings to %s: %s", path, e)


def apply_config_to_env(cfg: dict) -> None:
    """Translate the config into the env vars the server reads."""

    def set_or_unset(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    set_or_unset("MLX_VLM_PRELOAD_MODEL", cfg.get("model_name"))
    set_or_unset("MLX_VLM_DRAFT_MODEL", cfg.get("draft_model"))
    set_or_unset(
        "MLX_VLM_DRAFT_KIND",
        cfg.get("draft_kind") if cfg.get("draft_model") else None,
    )
    set_or_unset("MAX_KV_SIZE", cfg.get("max_context_length"))
    set_or_unset(
        "KV_BITS", DEFAULT_KV_QUANT_BITS if cfg.get("kv_quantization") else None
    )
    set_or_unset(AUTO_UNLOAD_TIME_ENV, cfg.get("auto_unload_time"))


def initialize_from_config() -> None:
    """Load the config file (if configured) and export it to the env.

    Called at server startup before the model preload, so the file — not
    command-line flags — decides which model to serve and with which
    settings.
    """
    cfg = load_config()
    if cfg is None:
        return
    logger.info(
        "Config: %s -> model=%s draft=%s max_context_length=%s "
        "kv_quantization=%s auto_unload_time=%s",
        config_path(),
        cfg.get("model_name"),
        cfg.get("draft_model"),
        cfg.get("max_context_length"),
        cfg.get("kv_quantization"),
        cfg.get("auto_unload_time"),
    )
    apply_config_to_env(cfg)
