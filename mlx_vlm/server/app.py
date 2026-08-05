import asyncio
import gc
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from threading import Lock, Thread
from types import SimpleNamespace
from typing import List, Optional, Tuple

import mlx.core as mx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound, RepositoryNotFoundError

from .. import apc as _apc
from ..generate.edit_image import load_image_edit_model
from ..generate.image import is_image_generation_model, load_image_generation_model
from ..structured import build_json_schema_logits_processor
from ..tool_parsers import _infer_tool_parser_from_processor
from ..version import __version__
from ..vision_cache import VisionFeatureCache
from . import request_normalization as _request_normalization
from .anthropic import register_routes as register_anthropic_routes
from .audio import register_routes as register_audio_routes
from .generation import (
    GenerationArguments,
    PromptTooLongError,
    ResponseGenerator,
    ServerMetricsStore,
    _build_metrics_envelope,
    get_configured_context_limit,
    get_kv_group_size,
    get_kv_quant_scheme,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_top_logprobs_k,
)
from .junie import register_control_routes, start_background_model_load
from .junie.lifecycle import lifecycle
from .openai import register_routes as register_openai_routes
from .responses_state import _split_thinking as _split_thinking_text
from .runtime import ModelCacheRegistry, runtime
from .schemas import ChatLogprobContent, ModelsResponse, TopLogprob

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
SERVER_API_KEY_ENV = "MLX_VLM_SERVER_API_KEY"

logger = logging.getLogger("mlx_vlm.server")

_as_plain_dict = _request_normalization._as_plain_dict


def _server_api_key() -> Optional[str]:
    key = os.environ.get(SERVER_API_KEY_ENV)
    return key if key else None


def _require_management_api_key(request: Request) -> None:
    api_key = _server_api_key()
    if api_key is None:
        return

    expected = f"Bearer {api_key}"
    supplied = request.headers.get("Authorization", "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _cache_group_for_cache(cache: dict) -> str:
    model_kind = cache.get("model_kind")
    if model_kind == "image_generation":
        return "image_generation"
    if model_kind == "image_edit":
        return "image_edit"
    if model_kind == "audio_tts":
        return "tts"
    if model_kind == "audio_stt":
        return "stt"
    if model_kind == "audio":
        return "audio"
    return "text_generation"


def _model_cache_registry() -> ModelCacheRegistry:
    cache = runtime.model_cache
    if isinstance(cache, ModelCacheRegistry):
        return cache

    registry = ModelCacheRegistry()
    if isinstance(cache, dict) and cache:
        registry.set(_cache_group_for_cache(cache), cache)
    runtime.model_cache = registry
    return registry


def _server_runtime_snapshot() -> dict:
    registry = _model_cache_registry()
    default_cache = registry.for_kind("text_generation")
    processor = default_cache.get("processor")
    config = default_cache.get("config")
    text_config = getattr(config, "text_config", None)
    native_context_size = getattr(text_config, "max_position_embeddings", None)
    configured_context_limit = get_configured_context_limit()
    effective_context_limit = (
        min(native_context_size, configured_context_limit)
        if native_context_size is not None and configured_context_limit is not None
        else configured_context_limit or native_context_size
    )
    queue_depth = 0
    if runtime.response_generator is not None and hasattr(
        runtime.response_generator, "requests"
    ):
        try:
            queue_depth = runtime.response_generator.requests.qsize()
        except Exception:
            queue_depth = 0
    audio_queue_depth = 0
    if runtime.audio_queue is not None and hasattr(runtime.audio_queue, "qsize"):
        try:
            audio_queue_depth = runtime.audio_queue.qsize()
        except Exception:
            audio_queue_depth = 0
    return {
        "loaded_model": default_cache.get("model_path", None),
        "loaded_adapter": default_cache.get("adapter_path", None),
        "loaded_models": {
            group: {
                "model": cache.get("model_path"),
                "adapter": cache.get("adapter_path"),
                "model_kind": cache.get("model_kind"),
            }
            for group, cache in registry.items()
        },
        "model_kind": default_cache.get("model_kind", "text_generation"),
        "loaded_context_size": native_context_size,
        "configured_context_limit": configured_context_limit,
        "effective_context_limit": effective_context_limit,
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(processor) if processor else None
        ),
        "continuous_batching_enabled": runtime.response_generator is not None,
        "request_queue_depth": queue_depth,
        "audio_queue_depth": audio_queue_depth,
        "apc": (
            {"enabled": False}
            if runtime.apc_manager is None
            else {"enabled": True, **runtime.apc_manager.stats_snapshot()}
        ),
    }


def _build_gen_args(
    request, processor=None, tenant_id: Optional[str] = None
) -> GenerationArguments:
    """Build GenerationArguments from a compatible API request."""
    return _request_normalization._build_gen_args(
        request,
        processor=processor,
        tenant_id=tenant_id,
        structured_logits_processor_builder=_build_structured_logits_processors,
    )


def _read_tenant_id(http_request) -> Optional[str]:
    """Pull a per-tenant APC salt from the request headers.

    Honoured headers (in order): ``X-APC-Tenant``, ``X-Tenant-Id``.
    """
    if http_request is None or not hasattr(http_request, "headers"):
        return None
    h = http_request.headers
    return h.get("x-apc-tenant") or h.get("x-tenant-id") or None


async def _preflight_stream_context_budget(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    images: Optional[List] = None,
    audio: Optional[List] = None,
    videos: Optional[List] = None,
    args: GenerationArguments,
):
    """Reject over-budget streaming requests before the HTTP stream starts."""
    if runtime.response_generator is None:
        return
    try:
        validate_kwargs = {"images": images, "audio": audio, "args": args}
        if videos is not None:
            validate_kwargs["videos"] = videos
        await asyncio.to_thread(
            runtime.response_generator.validate_context_budget,
            prompt,
            **validate_kwargs,
        )
    except PromptTooLongError as e:
        runtime.metrics.record_failure(
            endpoint=endpoint,
            model=model,
            stream=True,
            error=str(e),
        )
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=400, detail=str(e))


def _build_structured_logits_processors(request, processor):
    return _request_normalization._build_structured_logits_processors(
        request,
        processor,
        logits_processor_factory=_server_package_attr(
            "build_json_schema_logits_processor",
            build_json_schema_logits_processor,
        ),
    )


def _extract_response_format_schema(request):
    """Retain the package-level compatibility alias for schema extraction."""
    return _request_normalization._extract_response_format_schema(request)


def _count_thinking_tag_tokens(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    if (
        thinking_start_token
        and thinking_end_token
        and thinking_start_token in text
        and thinking_end_token in text
    ):
        return 2
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content)."""
    return _split_thinking_text(text, thinking_start_token, thinking_end_token)


def _decode_token(tokenizer, token_id: int) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Shared mutable server runtime state.
runtime.metrics = ServerMetricsStore()


def _server_package_attr(name, fallback=None):
    package = sys.modules.get(__package__)
    if package is not None and hasattr(package, name):
        return getattr(package, name)
    if fallback is not None:
        return fallback
    return globals()[name]


def __getattr__(name):
    legacy_runtime_attrs = {
        "model_cache": "model_cache",
        "response_generator": "response_generator",
        "apc_manager": "apc_manager",
        "server_metrics": "metrics",
    }
    if name in legacy_runtime_attrs:
        return getattr(runtime, legacy_runtime_attrs[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_audio_model(model_path: str):
    from mlx_audio.utils import load_model

    return load_model(model_path)


def _start_seed_prefix_warmup(on_done=None) -> bool:
    """Prefill and pin a stable cross-session prompt prefix at startup.

    ``MLX_VLM_SEED_REQUEST`` points at a chat-completions request body whose
    rendered prompt is a shared prefix of every future conversation (for
    Junie: the system message + tool schemas + first user message). The
    request is replayed against the server's own endpoint with
    ``max_tokens=1`` so it takes the exact same template/tokenize path as
    real traffic; the APC harvest then stores a session checkpoint just
    inside the stable region (the exact-prefix guard keeps it clear of the
    trailing generation header), and the pin keeps it from ever being
    LRU-evicted. With ``APC_DISK_PATH`` set the snapshot also persists, so
    later restarts warm from disk instead of re-prefilling.

    Returns True when a warmup thread was started; ``on_done`` (optional)
    fires when that thread finishes, successfully or not.
    """
    seed_path = os.environ.get("MLX_VLM_SEED_REQUEST")
    if not seed_path:
        return False

    port = os.environ.get("MLX_VLM_SERVER_PORT")
    if not port:
        logger.warning("Seed request set but server port unknown; skipping.")
        return False

    def warmup():
        import urllib.error
        import urllib.request

        try:
            with open(seed_path) as f:
                body = json.load(f)
        except Exception as e:
            logger.warning("Seed request: cannot read %s: %s", seed_path, e)
            return
        body["max_tokens"] = 1
        body["stream"] = False
        data = json.dumps(body).encode()
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        manager = runtime.apc_manager
        if manager is None or not callable(
            getattr(manager, "pin_next_session_store", None)
        ):
            logger.info("Seed request: APC not enabled; skipping warmup.")
            return
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("MLX_VLM_SERVER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        manager.pin_next_session_store()
        started = time.time()
        try:
            request = urllib.request.Request(
                f"{base}/v1/chat/completions", data=data, headers=headers
            )
            with urllib.request.urlopen(request, timeout=3600) as resp:
                payload = json.loads(resp.read())
            usage = payload.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            logger.info(
                "Seed prefix warmed and pinned: prompt_tokens=%s "
                "cached_tokens=%s elapsed=%.1fs source=%s",
                usage.get("prompt_tokens"),
                details.get("cached_tokens"),
                time.time() - started,
                seed_path,
            )
        except Exception as e:
            logger.warning("Seed prefix warmup failed: %s", e)

    def run():
        try:
            warmup()
        finally:
            if on_done is not None:
                on_done()

    Thread(target=run, daemon=True, name="apc-seed-warmup").start()
    return True


@asynccontextmanager
async def lifespan(app):
    dequant_prefill = os.environ.get("MLX_VLM_DEQUANT_PREFILL", "")
    if dequant_prefill.lower() in ("1", "true", "yes", "on"):
        from ..dequant_prefill import apply as _apply_dequant_prefill

        _apply_dequant_prefill()
        logger.info("Dequantize-on-the-fly prefill patch applied.")

    int8_prefill = os.environ.get("MLX_VLM_INT8_PREFILL", "")
    if int8_prefill.lower() in ("1", "true", "yes", "on"):
        from ..int8_prefill import apply as _apply_int8_prefill

        _apply_int8_prefill()
        logger.info("int8 NAX prefill patch applied.")

    # Model preload + seed warmup run on a background thread with lifecycle
    # phase tracking (see server/junie): the HTTP server is up immediately
    # and inference gets 503 until the model is in.
    start_background_model_load(_junie_deps)
    try:
        yield
    finally:
        if runtime.audio_queue is not None:
            runtime.audio_queue.stop_and_join()
            runtime.audio_queue = None


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)
inference_router = APIRouter(
    dependencies=[Depends(_require_management_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once


_INHERIT_ADAPTER = object()


def _unload_model_cache_group(cache_group: str) -> bool:
    registry = _model_cache_registry()
    cache = registry.for_kind(cache_group)
    if not cache:
        return False

    logger.info(
        "Unloading %s model: %s (adapter=%s)",
        cache_group,
        cache.get("model_path"),
        cache.get("adapter_path"),
    )

    response_generator = cache.get("response_generator")
    if response_generator is not None:
        logger.info("Stopping response generator.")
        response_generator.stop_and_join()
        if runtime.response_generator is response_generator:
            runtime.response_generator = None

    apc_manager = cache.get("apc_manager")
    if apc_manager is not None:
        apc_manager.clear()
        if runtime.apc_manager is apc_manager:
            runtime.apc_manager = None

    if "vision_cache" in cache:
        cache["vision_cache"].clear()

    registry.pop(cache_group)
    gc.collect()
    mx.clear_cache()
    return True


def _audio_model_kind(model_kind: str) -> bool:
    return model_kind in ("audio", "audio_tts", "audio_stt")


def _audio_cache_group(model_kind: str) -> str:
    if model_kind == "audio_tts":
        return "tts"
    if model_kind == "audio_stt":
        return "stt"
    return "audio"


def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    Also creates/updates the ResponseGenerator for continuous batching.
    """
    busy_phase = lifecycle.busy_phase_for_caller()
    if busy_phase is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model serving is unavailable: server phase is '{busy_phase}'. "
                "Poll GET /status and retry once the phase is 'ready'."
            ),
        )

    load_as_edit = model_kind == "image_edit"
    load_as_audio = _audio_model_kind(model_kind)
    load_as_image = model_kind == "image_generation" or (
        model_kind == "auto" and is_image_generation_model(model_path)
    )
    if load_as_edit:
        cache_group = "image_edit"
        effective_model_kind = "image_edit"
    elif load_as_audio:
        cache_group = _audio_cache_group(model_kind)
        effective_model_kind = model_kind
    elif load_as_image:
        cache_group = "image_generation"
        effective_model_kind = "image_generation"
    else:
        cache_group = "text_generation"
        effective_model_kind = "text_generation" if model_kind == "auto" else model_kind

    registry = _model_cache_registry()
    if adapter_path is _INHERIT_ADAPTER:
        cached_cache = registry.for_kind(cache_group)
        cached = cached_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path, effective_model_kind)
    cached_cache = registry.for_kind(cache_group)

    # Return from cache if already loaded and matches the requested paths
    if cached_cache and cached_cache.get("cache_key") == cache_key:
        if cache_group == "text_generation":
            runtime.response_generator = cached_cache.get("response_generator")
            runtime.apc_manager = cached_cache.get("apc_manager")
        logger.debug("Using cached model: %s (adapter=%s)", model_path, adapter_path)
        return (
            cached_cache["model"],
            cached_cache["processor"],
            cached_cache["config"],
        )

    # If this kind has a different model cached, clear only that cache group.
    if cached_cache:
        logger.info("New %s model requested; clearing its existing cache.", cache_group)
        _unload_model_cache_group(cache_group)

    if load_as_edit:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image edit models.",
            )
        logger.info("Loading image edit model: %s", model_path)
        try:
            model = load_image_edit_model(model_path)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image edit model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load image edit model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "family", "image_edit"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_edit",
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    if load_as_image:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image generation models.",
            )
        logger.info("Loading image generation model: %s", model_path)
        try:
            model = load_image_generation_model(model_path)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image generation model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load image generation model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "family", "image_generation"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_generation",
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    if load_as_audio:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for audio models.",
            )
        logger.info("Loading audio model: %s", model_path)
        try:
            model = _server_package_attr("load_audio_model", load_audio_model)(
                model_path
            )
        except RepositoryNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model not found: {model_path!r} is not a known "
                    "Hugging Face repo or local path"
                ),
            ) from e
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported audio model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load audio model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "model_type", "audio"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": model_kind,
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    vision_cache_size = int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20"))
    vision_cache = VisionFeatureCache(max_size=vision_cache_size)

    # KV cache quantization (uniform or TurboQuant)
    kv_bits = get_quantized_kv_bits(model_path)
    kv_group_size = get_kv_group_size()
    quantized_kv_start = get_quantized_kv_start()
    kv_quant_scheme = get_kv_quant_scheme()

    runtime.apc_manager = _apc.from_env(
        model_namespace=_apc.apc_disk_namespace(
            model_path,
            adapter_path=adapter_path,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_quant_scheme=kv_quant_scheme,
            quantized_kv_start=quantized_kv_start,
        )
    )

    response_generator = ResponseGenerator(
        model_path=model_path,
        adapter_path=adapter_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
        top_logprobs_k=get_top_logprobs_k(),
        apc_manager=runtime.apc_manager,
    )
    try:
        model, processor, config = response_generator.wait_until_ready()
    except Exception:
        response_generator.stop_and_join()
        vision_cache.clear()
        raise

    # Dry-run APC layout when the shared pool is enabled (log-only; never blocks serve).
    if runtime.apc_manager is not None:
        try:
            _apc.self_check_model_apc(model, kv_bits=kv_bits)
        except Exception as exc:
            logger.warning("APC self-check raised unexpectedly: %s", exc)

    cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
        "model_kind": "text_generation",
        "response_generator": response_generator,
        "apc_manager": runtime.apc_manager,
    }
    registry.set(cache_group, cache)
    runtime.response_generator = response_generator
    runtime.apc_manager = cache["apc_manager"]

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    unloaded_any = False
    if runtime.audio_queue is not None:
        is_audio_worker = getattr(
            runtime.audio_queue, "is_worker_thread", lambda: False
        )
        if not is_audio_worker():
            logger.info("Stopping audio request queue.")
            runtime.audio_queue.stop_and_join()
            runtime.audio_queue = None
            unloaded_any = True

    registry = _model_cache_registry()
    # Collect only the group NAMES: a list of items() tuples would keep every
    # cache dict — and the model weights — referenced until this function
    # returns, so the weights would only be freed after the clear_cache()
    # below and stay resident in the MLX buffer cache (GBs) indefinitely.
    for cache_group in [group for group, _ in registry.items()]:
        unloaded_any = _unload_model_cache_group(cache_group) or unloaded_any

    runtime.response_generator = None
    runtime.apc_manager = None
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    if unloaded_any:
        logger.info("Model caches cleared.")
    return unloaded_any


_protocol_deps = SimpleNamespace(
    INHERIT_ADAPTER=_INHERIT_ADAPTER,
    get_cached_model=lambda *args, **kwargs: _server_package_attr("get_cached_model")(
        *args, **kwargs
    ),
    generate=lambda *args, **kwargs: _server_package_attr("generate")(*args, **kwargs),
    stream_generate=lambda *args, **kwargs: _server_package_attr("stream_generate")(
        *args, **kwargs
    ),
    apply_chat_template=lambda *args, **kwargs: _server_package_attr(
        "apply_chat_template"
    )(*args, **kwargs),
    infer_tool_parser_from_processor=lambda *args, **kwargs: _server_package_attr(
        "_infer_tool_parser_from_processor"
    )(*args, **kwargs),
    load_tool_module=lambda *args, **kwargs: _server_package_attr("load_tool_module")(
        *args, **kwargs
    ),
    build_gen_args=_build_gen_args,
    read_tenant_id=_read_tenant_id,
    preflight_stream_context_budget=_preflight_stream_context_budget,
    as_plain_dict=_as_plain_dict,
    split_thinking=_split_thinking,
    count_thinking_tag_tokens=_count_thinking_tag_tokens,
    make_logprob_content=_make_logprob_content,
    build_metrics_envelope=_build_metrics_envelope,
)
register_anthropic_routes(inference_router, _protocol_deps)
register_openai_routes(inference_router, _protocol_deps)
register_audio_routes(inference_router, _protocol_deps)

_junie_deps = SimpleNamespace(
    require_management_api_key=_require_management_api_key,
    server_runtime_snapshot=_server_runtime_snapshot,
    model_cache_registry=_model_cache_registry,
    # Late-bound module globals so test monkeypatching keeps working.
    get_cached_model=lambda *args, **kwargs: get_cached_model(*args, **kwargs),
    unload_model_sync=lambda: unload_model_sync(),
    start_seed_prefix_warmup=lambda **kwargs: _start_seed_prefix_warmup(**kwargs),
)
register_control_routes(app, _junie_deps)


@inference_router.get("/models", response_model=ModelsResponse)
@inference_router.get(
    "/v1/models",
    response_model=ModelsResponse,
    include_in_schema=False,
)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    required_files = {"config.json", "tokenizer_config.json"}

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        has_weights = "model.safetensors.index.json" in file_names or any(
            file_name.endswith(".safetensors") for file_name in file_names
        )
        return required_files.issubset(file_names) and has_weights

    # Scan the cache directory for downloaded mlx models when it exists.
    try:
        hf_cache_info = _server_package_attr("scan_cache_dir", scan_cache_dir)()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]
    except CacheNotFound:
        downloaded_models = []

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]
    loaded_models = {
        cache.get("model_path")
        for cache in _model_cache_registry().values()
        if cache.get("model_path")
    }
    loaded_model = _model_cache_registry().get("model_path")
    if loaded_model:
        loaded_models.add(loaded_model)
    for loaded in sorted(loaded_models):
        if all(model["id"] != loaded for model in models):
            models.append(
                {"id": loaded, "object": "model", "created": int(time.time())}
            )

    response = {"object": "list", "data": models}

    return response


app.include_router(inference_router)


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"mlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check(request: Request):
    """
    Check if the server is healthy and what model is loaded.
    """
    _require_management_api_key(request)
    runtime = _server_runtime_snapshot()
    return {
        "status": "healthy",
        "loaded_model": runtime["loaded_model"],
        "loaded_adapter": runtime["loaded_adapter"],
        "loaded_models": runtime["loaded_models"],
        "loaded_context_size": runtime["loaded_context_size"],
        "configured_context_limit": runtime["configured_context_limit"],
        "effective_context_limit": runtime["effective_context_limit"],
        "loaded_tool_parser": runtime["loaded_tool_parser"],
        "continuous_batching_enabled": runtime["continuous_batching_enabled"],
        "apc_enabled": runtime["apc"]["enabled"],
    }


@app.get("/metrics")
@app.get("/v1/metrics", include_in_schema=False)
async def metrics_endpoint(request: Request):
    _require_management_api_key(request)
    payload = runtime.metrics.snapshot()
    payload["server"] = _server_runtime_snapshot()
    return payload


@app.get("/v1/cache/stats")
@app.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats(request: Request):
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    _require_management_api_key(request)
    if runtime.apc_manager is None:
        return {"enabled": False}
    snap = runtime.apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@app.post("/v1/cache/reset")
@app.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset(request: Request):
    _require_management_api_key(request)
    if runtime.apc_manager is None:
        return {"enabled": False}
    runtime.apc_manager.clear()
    # Return the freed KV buffers to the OS instead of leaving them parked
    # in the MLX buffer cache (same pattern as unload_model_sync).
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return {"enabled": True, "status": "cleared"}


@app.post("/unload")
async def unload_model_endpoint(request: Request):
    """
    Unload the currently loaded model from memory.
    """
    _require_management_api_key(request)
    snapshot = _server_runtime_snapshot()
    unloaded_info = {
        "model_name": snapshot["loaded_model"],
        "adapter_name": snapshot["loaded_adapter"],
        "models": snapshot["loaded_models"],
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
