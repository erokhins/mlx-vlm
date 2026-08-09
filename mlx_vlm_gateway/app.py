import argparse
import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional, Sequence

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from mlx_vlm_shared.errors import OUT_OF_MEMORY_ERROR_CODE
from mlx_vlm_shared.server_settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
)

from .settings import RESTART_SETTING_KEYS, SettingsStore, SettingsValidationError
from .supervisor import GatewaySettings, WorkerSupervisor, worker_command


logger = logging.getLogger("mlx_vlm.gateway")

WORKER_LOG_NAME = "junie-mlx-vlm.log"


def worker_connect_host(host: str) -> str:
    """The address to reach a worker bound to ``host`` from this process."""
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


async def _wait_for_disconnect(request: Request) -> None:
    """Resolve when the upstream client disconnects.

    Starlette's listen_for_disconnect pattern: the handler has already
    consumed the request body, so the only message left on the ASGI
    receive channel is ``http.disconnect``, making this an event-driven
    disconnect signal (no polling).
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _proxy_response(response: httpx.Response) -> Response:
    headers = {}
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=headers,
    )


def _is_confirmed_out_of_memory(response: httpx.Response) -> bool:
    if response.status_code != 503:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    return (
        isinstance(error, dict)
        and error.get("code") == OUT_OF_MEMORY_ERROR_CODE
    )


def create_app(
    settings: GatewaySettings,
    client_factory: Optional[Callable[[httpx.Timeout], httpx.AsyncClient]] = None,
    shutdown_callback: Optional[Callable[[], None]] = None,
) -> FastAPI:
    async def stop_idle_worker(app: FastAPI) -> None:
        while True:
            await asyncio.sleep(settings.idle_check_interval_s)
            sup = app.state.supervisor
            timeout_s = app.state.settings_store.current()["auto_unload_time"]
            if (
                timeout_s is None
                or sup.state != "ready"
                or sup.active_requests > 0
                or time.monotonic() - sup.last_activity_at < timeout_s
            ):
                continue
            lock = app.state.lifecycle_lock
            if lock.locked():
                continue
            async with lock:
                timeout_s = app.state.settings_store.current()["auto_unload_time"]
                if (
                    timeout_s is not None
                    and sup.state == "ready"
                    and sup.active_requests == 0
                    and time.monotonic() - sup.last_activity_at >= timeout_s
                ):
                    logger.info(
                        "Stopping inference worker after %ss without requests",
                        timeout_s,
                    )
                    await sup.stop_worker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            connect=2.0,
            read=settings.request_timeout_s,
            write=30.0,
            pool=5.0,
        )
        client = (
            client_factory(timeout)
            if client_factory is not None
            else httpx.AsyncClient(timeout=timeout)
        )
        supervisor = WorkerSupervisor(settings, client)
        app.state.client = client
        app.state.supervisor = supervisor
        app.state.settings_store = SettingsStore(settings.config_path)
        app.state.lifecycle_lock = asyncio.Lock()
        app.state.shutting_down = False
        app.state.models_payload = None
        await supervisor.open()
        idle_task = asyncio.create_task(
            stop_idle_worker(app),
            name="mlx-vlm-idle-worker-stop",
        )
        try:
            yield
        finally:
            idle_task.cancel()
            try:
                await idle_task
            except asyncio.CancelledError:
                pass
            await supervisor.close()
            await client.aclose()

    app = FastAPI(title="MLX-VLM Gateway", lifespan=lifespan)

    def supervisor(request: Request) -> WorkerSupervisor:
        return request.app.state.supervisor

    def settings_store(request: Request) -> SettingsStore:
        return request.app.state.settings_store

    def status_payload(request: Request) -> dict:
        sup = supervisor(request)
        current = settings_store(request).current()
        if request.app.state.shutting_down:
            phase = "stopping"
        else:
            phase = {
                "starting": "loading_model",
                "restarting": "restarting",
                "stopping": "stopping",
                "error": "error",
            }.get(sup.state, "ready")
        model_id = sup.worker_health.get("loaded_model") or current["model_name"]
        return {
            "phase": phase,
            "phase_detail": (
                sup.last_error if phase in {"restarting", "error"} else None
            ),
            "phase_since_unix": sup.state_since_unix,
            "uptime_s": round(max(0.0, time.monotonic() - sup._started_at), 3),
            "model": {
                "loaded": sup.state == "ready" and bool(sup.process),
                "id": model_id,
                "draft_model": settings_store(request).draft_model(),
                "context_limit": current["max_context_length"],
            },
            "memory": sup.worker_health.get("memory") or {},
            "inference": {
                "in_progress": sup.active_requests > 0,
                "in_flight": sup.active_requests,
                "queue_depth": max(0, sup.active_requests - 1),
                "requests": [],
            },
        }

    @app.get("/status")
    @app.get("/v1/status", include_in_schema=False)
    async def status(request: Request):
        return status_payload(request)

    @app.get("/current_settings")
    @app.get("/v1/current_settings", include_in_schema=False)
    async def current_settings(request: Request):
        return settings_store(request).current()

    @app.post("/apply_settings")
    @app.post("/v1/apply_settings", include_in_schema=False)
    async def apply_settings(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Request body must be a JSON object.",
            ) from exc
        store = settings_store(request)
        try:
            updates, force = store.validate(body)
        except SettingsValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        lock = request.app.state.lifecycle_lock
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Another settings change is already in progress.",
            )

        async with lock:
            sup = supervisor(request)
            current = store.current()
            updates = {
                key: value
                for key, value in updates.items()
                if current.get(key) != value
            }
            if not updates:
                return {
                    "status": "applied",
                    "changes": [],
                    "settings": current,
                }

            def persist_updates() -> dict:
                try:
                    return store.save(updates)
                except OSError as exc:
                    logger.error("Failed to save settings: %s", exc)
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to save settings; worker state was not changed.",
                    ) from exc

            restart_changes = set(updates) & RESTART_SETTING_KEYS
            if not restart_changes:
                return {
                    "status": "applied",
                    "changes": sorted(updates),
                    "settings": persist_updates(),
                }

            if sup.state in {"starting", "restarting", "stopping"}:
                phase = status_payload(request)["phase"]
                raise HTTPException(
                    status_code=409,
                    detail=f"Server is busy (phase '{phase}'); retry once it settles.",
                )

            if sup.active_requests > 0 and not force:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{sup.active_requests} inference request(s) in flight; pass "
                        '"force": true to restart model serving anyway.'
                    ),
                )

            target = {**current, **updates}
            persist_updates()
            await sup.stop_worker()
            await sup.start_worker(wait_ready=False)
            return {
                "status": "applying",
                "model": target["model_name"],
                "changes": sorted(updates),
                "message": (
                    "Model serving is restarting; poll GET /status until "
                    "phase is 'ready'."
                ),
            }

    @app.get("/health")
    async def health(request: Request):
        sup = supervisor(request)
        if sup.state == "ready":
            try:
                response = await request.app.state.client.get(
                    f"{settings.worker_url}/health",
                    timeout=settings.probe_timeout_s,
                )
                if response.status_code == 200:
                    return JSONResponse(response.json())
            except (httpx.RequestError, ValueError):
                pass
        current = settings_store(request).current()
        return {
            "status": "healthy",
            "loaded_model": None,
            "loaded_adapter": None,
            "loaded_models": {},
            "loaded_context_size": None,
            "configured_context_limit": current["max_context_length"],
            "effective_context_limit": None,
            "loaded_tool_parser": None,
            "continuous_batching_enabled": False,
            "apc_enabled": False,
        }

    @app.get("/ready")
    async def ready(request: Request):
        state = supervisor(request).snapshot()
        status_code = 200 if state["status"] == "ready" else 503
        return JSONResponse({"status": state["status"], "worker": state}, status_code)

    async def management_proxy(request: Request, method: str, path: str):
        sup = supervisor(request)
        if sup.state != "ready":
            raise HTTPException(status_code=503, detail="Inference worker is not ready")
        worker_generation = sup.generation
        try:
            response = await request.app.state.client.request(
                method, f"{settings.worker_url}{path}"
            )
        except httpx.RequestError as exc:
            sup.schedule_restart(
                f"worker connection failed: {exc}",
                generation=worker_generation,
            )
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted; please retry",
            ) from exc
        return response

    @app.get("/models")
    @app.get("/v1/models", include_in_schema=False)
    async def models(request: Request):
        if supervisor(request).state == "ready":
            response = await management_proxy(request, "GET", "/v1/models")
            if response.status_code == 200:
                try:
                    request.app.state.models_payload = response.json()
                except ValueError:
                    pass
            return _proxy_response(response)

        if request.app.state.models_payload is not None:
            return request.app.state.models_payload

        store = settings_store(request)
        model_ids = [store.draft_model(), store.current()["model_name"]]
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0}
                for model_id in dict.fromkeys(model_ids)
                if model_id
            ],
        }

    @app.post("/unload")
    @app.post("/v1/unload", include_in_schema=False)
    async def unload(request: Request):
        sup = supervisor(request)
        async with request.app.state.lifecycle_lock:
            process = sup.process
            if process is None or process.returncode is not None:
                await sup.stop_worker()
                return {
                    "status": "no_model_loaded",
                    "message": "No model is currently loaded",
                }
            worker = dict(sup.worker_health)
            unloaded = {
                "model_name": worker.get("loaded_model")
                or settings_store(request).current()["model_name"],
                "adapter_name": worker.get("loaded_adapter"),
                "models": worker.get("loaded_models", {}),
            }
            await sup.stop_worker()
        return {
            "status": "success",
            "message": "Model unloaded successfully",
            "unloaded": unloaded,
        }

    @app.get("/metrics")
    @app.get("/v1/metrics", include_in_schema=False)
    async def metrics(request: Request):
        response = await management_proxy(request, "GET", "/metrics")
        try:
            payload = response.json()
        except ValueError:
            return _proxy_response(response)
        payload["gateway"] = supervisor(request).metrics()
        return JSONResponse(payload, status_code=response.status_code)

    @app.get("/cache/stats")
    @app.get("/v1/cache/stats", include_in_schema=False)
    async def cache_stats(request: Request):
        return _proxy_response(await management_proxy(request, "GET", "/cache/stats"))

    @app.post("/shutdown")
    @app.post("/v1/shutdown", include_in_schema=False)
    async def shutdown(request: Request):
        if request.app.state.shutting_down:
            return {"status": "shutting_down"}
        request.app.state.shutting_down = True
        async with request.app.state.lifecycle_lock:
            await supervisor(request).stop_worker()

        callback = shutdown_callback
        if callback is None:

            def callback():
                os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_running_loop().call_later(0.05, callback)
        return {"status": "shutting_down"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        sup = supervisor(request)
        if request.app.state.shutting_down:
            raise HTTPException(
                status_code=503,
                detail="Model serving is unavailable: server phase is 'stopping'.",
            )
        if sup.state == "error" and not sup.startup_retry_due():
            raise HTTPException(
                status_code=503,
                detail="Inference worker is recovering; please retry shortly.",
            )
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Request body must be JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="Request body must be a JSON object"
            )
        payload["stream"] = False
        headers = {"content-type": "application/json"}
        for name in ("x-apc-tenant", "x-tenant-id"):
            if value := request.headers.get(name):
                headers[name] = value

        sup.requests_forwarded += 1
        sup.active_requests += 1
        sup.last_activity_at = time.monotonic()
        try:
            try:
                async with request.app.state.lifecycle_lock:
                    if request.app.state.shutting_down:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "Model serving is unavailable: server phase is "
                                "'stopping'."
                            ),
                        )
                    if sup.state != "ready":
                        await sup.start_worker(wait_ready=False)

                if sup.state != "ready":
                    await sup.wait_until_ready()
            except RuntimeError as exc:
                sup.requests_failed += 1
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Inference worker did not become ready in time; please retry"
                    ),
                ) from exc
            worker_generation = sup.generation

            post_task = asyncio.create_task(
                request.app.state.client.post(
                    f"{settings.worker_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=settings.request_timeout_s,
                )
            )
            disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
            try:
                await asyncio.wait(
                    {post_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                disconnect_task.cancel()
            if not post_task.done():
                # The upstream client is gone. Aborting our connection is
                # what tells the worker to cancel the generation itself.
                post_task.cancel()
                try:
                    await post_task
                except (asyncio.CancelledError, httpx.HTTPError):
                    pass
                sup.requests_cancelled += 1
                logger.info(
                    "Client disconnected; aborted in-flight worker request."
                )
                # 499: client closed request; nobody reads this.
                return Response(status_code=499)
            response = post_task.result()
        except httpx.TimeoutException as exc:
            sup.requests_failed += 1
            sup.schedule_restart(
                "worker exceeded hard request timeout",
                generation=worker_generation,
            )
            raise HTTPException(
                status_code=504, detail="Inference worker did not stop in time"
            ) from exc
        except httpx.RequestError as exc:
            sup.requests_failed += 1
            sup.schedule_restart(
                f"worker connection failed during inference: {exc}",
                generation=worker_generation,
            )
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted; please retry",
            ) from exc
        finally:
            sup.active_requests = max(0, sup.active_requests - 1)
            sup.last_activity_at = time.monotonic()

        if _is_confirmed_out_of_memory(response):
            sup.requests_failed += 1
            sup.schedule_restart(
                "worker reported confirmed out-of-memory error",
                generation=worker_generation,
            )
            return _proxy_response(response)
        if response.status_code == 508:
            # The worker's private corrupted-generation signal (token-id-0
            # loop): restart it immediately and tell the client to retry —
            # the retry lands on a freshly started worker.
            sup.requests_failed += 1
            sup.schedule_restart(
                "worker reported corrupted generation (HTTP 508)",
                generation=worker_generation,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Generation produced corrupted output; model serving is "
                    "restarting. Retry shortly."
                ),
            )
        if sup.record_worker_response(
            response.status_code,
            generation=worker_generation,
        ):
            sup.requests_failed += 1
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted after repeated internal errors; please retry",
            )
        if response.status_code >= 400:
            sup.requests_failed += 1
        else:
            sup.requests_completed += 1
        return _proxy_response(response)

    return app


def worker_log_path(config_file: str) -> str:
    """The worker's log, beside the config that named the model it serves."""
    return os.path.join(os.path.dirname(config_file), WORKER_LOG_NAME)


def rotate_worker_log(path: str) -> None:
    """Start this run's log fresh, keeping the last one as <name>.0.

    One generation is enough to answer "it died, what happened before I
    restarted it", and it bounds what an unattended machine accumulates --
    a long run is not bounded, and log_raw_tokens writes every token.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.replace(path, f"{path}.0")
    except FileNotFoundError:
        pass


def apply_model_cache_env(config: dict) -> None:
    """Point the worker's Hugging Face cache at the configured models dir.

    Applied to this process because the worker inherits its environment,
    and it has to be: ``huggingface_hub`` reads both values once, when it
    is imported, which happens before the worker's own code runs.
    """
    os.environ["HF_HUB_CACHE"] = os.path.expanduser(config["models_dir"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_settings(path: str, config: dict) -> GatewaySettings:
    """Turn the config file into the daemon's launch-time settings."""
    if config["port"] == config["worker_port"]:
        raise SystemExit(
            f"ERROR: public port {config['port']} and worker port "
            f"{config['worker_port']} must differ; fix {path}."
        )
    worker_host = worker_connect_host(config["host"])
    return GatewaySettings(
        worker_url=f"http://{worker_host}:{config['worker_port']}",
        worker_command=worker_command(),
        worker_log_path=worker_log_path(path),
        config_path=path,
        startup_timeout_s=config["startup_timeout_s"],
        request_timeout_s=config["request_timeout_s"],
        startup_probe_interval_s=config["startup_probe_interval_s"],
        probe_interval_s=config["probe_interval_s"],
        probe_timeout_s=config["probe_timeout_s"],
        probe_failures_before_restart=config["probe_failures_before_restart"],
        max_start_failures=config["max_start_failures"],
        startup_retry_cooldown_s=config["startup_retry_cooldown_s"],
        restart_delay_s=config["restart_delay_s"],
        shutdown_timeout_s=config["shutdown_timeout_s"],
        idle_check_interval_s=config["idle_check_interval_s"],
    )


def main(argv: Optional[Sequence[str]] = None):
    argparse.ArgumentParser(
        description=(
            "MLX-VLM daemon: serves the public API and supervises the "
            "inference worker. Takes no arguments; every setting comes from "
            f"the config file (${CONFIG_PATH_ENV}, default "
            f"{DEFAULT_CONFIG_PATH})."
        )
    ).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    path = config_path()
    config = load_config(path)
    apply_model_cache_env(config)
    settings = build_settings(path, config)
    rotate_worker_log(settings.worker_log_path)
    logger.info("Config: %s (models: %s)", path, os.environ["HF_HUB_CACHE"])
    logger.info(
        "Worker log: %s (previous run kept as %s.0)",
        settings.worker_log_path,
        settings.worker_log_path,
    )
    uvicorn.run(
        create_app(settings),
        host=config["host"],
        port=config["port"],
        workers=1,
        server_header=False,
        log_level="info",
    )
