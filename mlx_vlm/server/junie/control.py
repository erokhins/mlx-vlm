"""Junie-local control plane for the server.

Keeps the Junie mac-app integration out of the core server modules:
background model (re)loading with lifecycle phases, the /status,
/current_settings, /apply_settings and /shutdown endpoints, and the live
inference-progress snapshot read from the batching engine.

Settings changes restart model serving (unload + reload with the new
environment) without restarting the HTTP server, so the port and these
endpoints stay available throughout; clients poll GET /status until the
phase is "ready".

``deps`` mirrors the register_routes(app, deps) pattern of the protocol
modules; app.py wires in its own helpers (get_cached_model,
unload_model_sync, the runtime snapshot, the seed warmup, ...).
"""

import logging
import os
import signal
import threading
import time
from threading import Thread, Timer

from fastapi import HTTPException, Request

from ..generation import get_configured_context_limit
from ..runtime import runtime
from .lifecycle import (
    PHASE_ERROR,
    PHASE_LOADING_MODEL,
    PHASE_READY,
    PHASE_RESTARTING,
    PHASE_WARMING_UP,
    lifecycle,
)
from .state import metrics_in_flight, reload_lock, serving_config
from .watchdog import (
    AUTO_UNLOAD_TIME_ENV,
    auto_unload_seconds,
    ensure_idle_watchdog,
)

logger = logging.getLogger("mlx_vlm.server")

DEFAULT_KV_QUANT_BITS = 8

_ALLOWED_SETTINGS_KEYS = {
    "model_name",
    "max_context_length",
    "kv_quantization",
    "auto_unload_time",
    "force",
}

# Env updates that require a model-serving restart to take effect;
# everything else (auto_unload_time) is applied live.
_RESTART_ENV_KEYS = {"MAX_KV_SIZE", "KV_BITS"}


# --------------------------------------------------------------------------
# Startup: background model load + seed warmup, with lifecycle phases.
# --------------------------------------------------------------------------


def start_background_model_load(deps) -> None:
    """Called from the lifespan instead of loading models inline.

    Loading on a background thread keeps the HTTP server (health/status/
    settings endpoints) responsive while the multi-GB weights load;
    get_cached_model() rejects inference from other threads with 503 until
    this finishes (see lifecycle.busy_phase_for_caller).
    """
    has_preload = any(
        os.environ.get(name)
        for name in (
            "MLX_VLM_PRELOAD_MODEL",
            "MLX_VLM_PRELOAD_IMAGE_MODEL",
            "MLX_VLM_PRELOAD_TTS_MODEL",
            "MLX_VLM_PRELOAD_STT_MODEL",
        )
    )
    if has_preload:
        lifecycle.set_phase(PHASE_LOADING_MODEL)
        Thread(
            target=_load_configured_models,
            args=(deps,),
            daemon=True,
            name="model-preload",
        ).start()
    else:
        lifecycle.set_phase(PHASE_READY)
    ensure_idle_watchdog(deps)


def _load_configured_models(deps) -> None:
    lifecycle.set_loader_thread(threading.get_ident())
    try:
        model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
        adapter_path = os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
        if model_path:
            serving_config["model_path"] = model_path
            serving_config["adapter_path"] = adapter_path
            lifecycle.set_phase(PHASE_LOADING_MODEL, f"loading {model_path}")
            logger.info("Pre-loading language model: %s", model_path)
            deps.get_cached_model(
                model_path, adapter_path, model_kind="text_generation"
            )
            serving_config["loaded_at"] = time.time()
            kv_bits = os.environ.get("KV_BITS")
            kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
            if kv_bits:
                logger.info(
                    "KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme
                )
            logger.info("Language model ready, continuous batching enabled.")

        preload_models = (
            (
                os.environ.pop("MLX_VLM_PRELOAD_IMAGE_MODEL", None),
                None,
                "image_generation",
                "image generation model",
            ),
            (
                os.environ.pop("MLX_VLM_PRELOAD_TTS_MODEL", None),
                None,
                "audio_tts",
                "text-to-speech model",
            ),
            (
                os.environ.pop("MLX_VLM_PRELOAD_STT_MODEL", None),
                None,
                "audio_stt",
                "speech-to-text model",
            ),
        )
        for preload_model_path, preload_adapter_path, model_kind, label in (
            preload_models
        ):
            if not preload_model_path:
                continue
            lifecycle.set_phase(PHASE_LOADING_MODEL, f"loading {preload_model_path}")
            logger.info("Pre-loading %s: %s", label, preload_model_path)
            deps.get_cached_model(
                preload_model_path,
                preload_adapter_path,
                model_kind=model_kind,
            )
            logger.info("%s ready.", label.capitalize())
    except Exception as e:
        logger.exception("Startup model load failed")
        lifecycle.set_phase(PHASE_ERROR, f"model load failed: {e}")
        return
    finally:
        lifecycle.set_loader_thread(None)
    _finish_model_startup(deps)


def _finish_model_startup(deps) -> None:
    """Kick the seed warmup when configured; otherwise the model is ready."""
    lifecycle.set_phase(PHASE_WARMING_UP, "prefilling pinned seed prompt")
    started = deps.start_seed_prefix_warmup(
        # Warmup is best-effort: whatever happened, the model itself is
        # loaded and serving.
        on_done=lambda: lifecycle.transition(PHASE_WARMING_UP, PHASE_READY)
    )
    if not started:
        lifecycle.set_phase(PHASE_READY)


# --------------------------------------------------------------------------
# Live inference progress.
# --------------------------------------------------------------------------


def _progress_snapshot(generator) -> list:
    """Read-only view of the generator's in-flight batched requests.

    Runs on request threads while the generation thread mutates the
    underlying dicts, so it copies defensively and tolerates entries
    vanishing mid-iteration.
    """
    now = time.perf_counter()
    snapshot = []
    for uid, info in list(getattr(generator, "_active_requests", {}).items()):
        try:
            prompt_tokens = int(info.get("prompt_tokens", 0) or 0)
            prefill_processed = int(info.get("prefill_processed", -1))
            generated_tokens = int(info.get("generated_tokens", 0) or 0)
            if info.get("decode_started_at") is not None:
                stage = "decode"
            elif prefill_processed >= 0:
                stage = "prefill"
            else:
                stage = "queued"
            entry = {
                "request_id": str(info.get("request_id", uid)),
                "stage": stage,
                "prompt_tokens": prompt_tokens,
                "prefill_processed": max(0, prefill_processed),
                "generated_tokens": generated_tokens,
            }
            if prompt_tokens > 0:
                entry["prefill_progress"] = round(
                    min(1.0, max(0, prefill_processed) / prompt_tokens), 4
                )
            queued_at = info.get("queued_at")
            if queued_at is not None:
                entry["elapsed_s"] = round(max(0.0, now - float(queued_at)), 3)
            snapshot.append(entry)
        except Exception:
            continue
    return snapshot


# --------------------------------------------------------------------------
# Settings.
# --------------------------------------------------------------------------


def _current_settings_payload(deps) -> dict:
    cache = deps.model_cache_registry().for_kind("text_generation")
    return {
        "model_name": cache.get("model_path") or serving_config["model_path"],
        "max_context_length": get_configured_context_limit(),
        "kv_quantization": bool(os.environ.get("KV_BITS")),
        "auto_unload_time": auto_unload_seconds(),
    }


def _settings_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _validate_settings(body: dict):
    """Translate an apply_settings body into (model_path, env updates).

    Env updates map env var name -> string value, or None to unset.
    """
    unknown = set(body) - _ALLOWED_SETTINGS_KEYS
    if unknown:
        raise _settings_error(f"Unknown settings: {sorted(unknown)}")

    env_updates = {}
    model_path = None

    if "model_name" in body:
        model_path = body["model_name"]
        if not isinstance(model_path, str) or not model_path.strip():
            raise _settings_error('"model_name" must be a non-empty string.')
        model_path = model_path.strip()

    for key, env_name in (
        ("max_context_length", "MAX_KV_SIZE"),
        ("auto_unload_time", AUTO_UNLOAD_TIME_ENV),
    ):
        if key not in body:
            continue
        value = body[key]
        if value is None:
            env_updates[env_name] = None
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            env_updates[env_name] = str(value)
        else:
            raise _settings_error(f'"{key}" must be a positive integer or null.')

    if "kv_quantization" in body:
        enabled = body["kv_quantization"]
        if not isinstance(enabled, bool):
            raise _settings_error('"kv_quantization" must be a boolean.')
        env_updates["KV_BITS"] = str(DEFAULT_KV_QUANT_BITS) if enabled else None

    return model_path, env_updates


def _apply_env(env_updates: dict) -> None:
    for key, value in env_updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _apply_settings_worker(deps, model_path, adapter_path, env_updates) -> None:
    lifecycle.set_loader_thread(threading.get_ident())
    try:
        logger.info("Applying settings: %s (model=%s)", env_updates, model_path)
        deps.unload_model_sync()
        _apply_env(env_updates)
        if model_path:
            lifecycle.set_phase(PHASE_LOADING_MODEL, f"loading {model_path}")
            deps.get_cached_model(
                model_path, adapter_path, model_kind="text_generation"
            )
            serving_config["model_path"] = model_path
            serving_config["adapter_path"] = adapter_path
            serving_config["loaded_at"] = time.time()
    except Exception as e:
        logger.exception("apply_settings reload failed")
        lifecycle.set_phase(PHASE_ERROR, f"apply_settings failed: {e}")
        return
    finally:
        lifecycle.set_loader_thread(None)
        reload_lock.release()
    if model_path:
        _finish_model_startup(deps)
    else:
        lifecycle.set_phase(PHASE_READY)


# --------------------------------------------------------------------------
# Endpoints.
# --------------------------------------------------------------------------


def register_control_routes(app, deps) -> None:
    @app.get("/status")
    @app.get("/v1/status", include_in_schema=False)
    async def status_endpoint(request: Request):
        """Server lifecycle phase plus live inference progress."""
        deps.require_management_api_key(request)
        phase = lifecycle.snapshot()
        snapshot = deps.server_runtime_snapshot()
        generator = runtime.response_generator
        progress = _progress_snapshot(generator) if generator is not None else []
        in_flight = metrics_in_flight()
        return {
            "phase": phase["phase"],
            "phase_detail": phase["detail"],
            "phase_since_unix": phase["since_unix"],
            "uptime_s": round(
                max(0.0, time.time() - lifecycle.process_started_at), 3
            ),
            "model": {
                "loaded": snapshot["loaded_model"] is not None,
                "id": snapshot["loaded_model"] or serving_config["model_path"],
                "draft_model": os.environ.get("MLX_VLM_DRAFT_MODEL"),
                "context_limit": snapshot["effective_context_limit"],
            },
            "inference": {
                "in_progress": in_flight > 0 or bool(progress),
                "in_flight": in_flight,
                "queue_depth": snapshot["request_queue_depth"],
                "requests": progress,
            },
        }

    @app.get("/current_settings")
    @app.get("/v1/current_settings", include_in_schema=False)
    async def current_settings_endpoint(request: Request):
        """The settings model serving is currently running with."""
        deps.require_management_api_key(request)
        return _current_settings_payload(deps)

    @app.post("/apply_settings")
    @app.post("/v1/apply_settings", include_in_schema=False)
    async def apply_settings_endpoint(request: Request):
        """Apply new serving settings. Accepts any subset of: model_name,
        max_context_length, kv_quantization, auto_unload_time; plus
        force=true to interrupt in-flight inference. auto_unload_time
        applies live; the others restart model serving (not the HTTP
        server) — poll GET /status until phase is "ready".
        """
        deps.require_management_api_key(request)
        try:
            body = await request.json()
        except Exception:
            raise _settings_error("Request body must be a JSON object.")
        if not isinstance(body, dict):
            raise _settings_error("Request body must be a JSON object.")
        requested_model, env_updates = _validate_settings(body)
        if requested_model is None and not env_updates:
            raise _settings_error("No settings provided.")
        force = bool(body.get("force"))
        needs_restart = requested_model is not None or any(
            key in _RESTART_ENV_KEYS for key in env_updates
        )

        if not reload_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="Another settings change is already in progress.",
            )

        if not needs_restart:
            # auto_unload_time only: applies live, nothing to restart.
            try:
                _apply_env(env_updates)
            finally:
                reload_lock.release()
            return {
                "status": "applied",
                "changes": sorted(set(body) - {"force"}),
                "settings": _current_settings_payload(deps),
            }

        try:
            phase = lifecycle.phase()
            if phase in (PHASE_LOADING_MODEL, PHASE_RESTARTING):
                raise HTTPException(
                    status_code=409,
                    detail=f"Server is busy (phase '{phase}'); retry once it settles.",
                )
            in_flight = metrics_in_flight()
            if in_flight > 0 and not force:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{in_flight} inference request(s) in flight; pass "
                        '"force": true to restart model serving anyway.'
                    ),
                )
            cache = deps.model_cache_registry().for_kind("text_generation")
            if requested_model:
                target_model = requested_model
                adapter_path = None
            else:
                target_model = (
                    cache.get("model_path") or serving_config["model_path"]
                )
                adapter_path = (
                    cache.get("adapter_path") or serving_config["adapter_path"]
                )
            lifecycle.set_phase(PHASE_RESTARTING, "applying new settings")
            Thread(
                target=_apply_settings_worker,
                args=(deps, target_model, adapter_path, env_updates),
                daemon=True,
                name="apply-settings",
            ).start()
        except Exception:
            reload_lock.release()
            raise
        return {
            "status": "applying",
            "model": target_model,
            "changes": sorted(set(body) - {"force"}),
            "message": (
                "Model serving is restarting; poll GET /status until "
                "phase is 'ready'."
            ),
        }

    @app.post("/shutdown")
    @app.post("/v1/shutdown", include_in_schema=False)
    async def shutdown_endpoint(request: Request):
        """Gracefully shut the whole server process down."""
        deps.require_management_api_key(request)
        logger.info("Shutdown requested via POST /shutdown.")
        Timer(0.5, os.kill, args=(os.getpid(), signal.SIGTERM)).start()
        return {"status": "shutting_down"}
