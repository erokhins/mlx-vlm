import argparse
import logging
import os

import uvicorn

from ..generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
)
from .generation import (
    DEFAULT_ENABLE_THINKING,
    get_log_progress_interval,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
)

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8085

logger = logging.getLogger("mlx_vlm.server")


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: 8085)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a language model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Pre-load an image generation model at startup.",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default=None,
        help="Pre-load a text-to-speech model at startup.",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=None,
        help="Pre-load a speech-to-text model at startup.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--dequant-prefill",
        action="store_true",
        help=(
            "Transiently dequantize quantized weights to bf16 for prefill-sized "
            "matmuls and use plain GEMMs (decode keeps the quantized kernels). "
            "Maps to the MLX_VLM_DEQUANT_PREFILL env var."
        ),
    )
    parser.add_argument(
        "--hybrid-fp16",
        action="store_true",
        help=(
            "Run attention/MoE block internals in fp16 for the early layers "
            "while keeping the residual stream bf16 (faster on M1-family GPUs, "
            "which emulate bfloat). Laguna models only. "
            "Maps to the MLX_VLM_HYBRID_FP16 env var."
        ),
    )
    parser.add_argument(
        "--int8-prefill",
        action="store_true",
        help=(
            "Route prefill-sized MLP matmuls through W8A8 int8 GEMMs on the "
            "M5 GPU neural accelerators (decode keeps the quantized kernels). "
            "Requires an M5-class GPU. Maps to the MLX_VLM_INT8_PREFILL env var."
        ),
    )
    parser.add_argument(
        "--log-progress-interval",
        type=int,
        default=get_log_progress_interval(),
        help=(
            "Decoded tokens between progress log messages; 0 disables periodic "
            "decode progress (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--preserve-thinking",
        action="store_true",
        help=(
            "Always render <think> blocks for assistant history turns "
            "(templates that support preserve_thinking, e.g. Qwen3 family). "
            "Keeps multi-turn prompt rendering position-independent so APC "
            "prefix caching survives new user turns. "
            "Maps to the MLX_VLM_PRESERVE_THINKING env var."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=get_server_thinking_budget(),
        help=(
            "Default maximum number of tokens allowed inside a thinking block. "
            "Requests can override this with thinking_budget."
        ),
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=get_server_thinking_start_token(),
        help=(
            "Default token that opens a thinking block. Requests can override "
            "this with thinking_start_token."
        ),
    )
    parser.add_argument(
        "--thinking-end-token",
        "--thinking-eos-token",
        dest="thinking_end_token",
        type=str,
        default=get_server_thinking_end_token(),
        help=(
            "Default token that closes a thinking block. Requests can override "
            "this with thinking_end_token."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family -- 'dflash', 'eagle3', or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--log-raw-tokens",
        action="store_true",
        help=(
            "Log each request's generated tokens as text once it finishes, "
            "with accepted speculative tokens colored (green = MTP drafter, "
            "cyan = n-gram prompt lookup)."
        ),
    )
    parser.add_argument(
        "--seed-request",
        type=str,
        default=None,
        help=(
            "Path to a chat-completions request body whose rendered prompt "
            "is a stable prefix of future conversations. It is prefilled at "
            "startup and pinned in APC so new sessions warm-start; with "
            "APC_DISK_PATH set the snapshot persists across restarts."
        ),
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Optional bearer token required for inference, model discovery, and "
            "management endpoints. "
            "Maps to the MLX_VLM_SERVER_API_KEY env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.image_model:
        os.environ["MLX_VLM_PRELOAD_IMAGE_MODEL"] = args.image_model
    if args.tts_model:
        os.environ["MLX_VLM_PRELOAD_TTS_MODEL"] = args.tts_model
    if args.stt_model:
        os.environ["MLX_VLM_PRELOAD_STT_MODEL"] = args.stt_model
    os.environ["MLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["MLX_VLM_DRAFT_MODEL"] = args.draft_model
    if args.draft_kind is not None:
        os.environ["MLX_VLM_DRAFT_KIND"] = args.draft_kind
    if args.draft_block_size is not None:
        os.environ["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    if args.dequant_prefill:
        os.environ["MLX_VLM_DEQUANT_PREFILL"] = "1"
    if args.int8_prefill:
        os.environ["MLX_VLM_INT8_PREFILL"] = "1"
    if args.hybrid_fp16:
        os.environ["MLX_VLM_HYBRID_FP16"] = "1"
    os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = str(args.log_progress_interval)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.preserve_thinking:
        os.environ["MLX_VLM_PRESERVE_THINKING"] = "1"
    if args.thinking_budget is not None:
        os.environ["MLX_VLM_THINKING_BUDGET"] = str(args.thinking_budget)
    if args.thinking_start_token is not None:
        os.environ["MLX_VLM_THINKING_START_TOKEN"] = args.thinking_start_token
    if args.thinking_end_token is not None:
        os.environ["MLX_VLM_THINKING_END_TOKEN"] = args.thinking_end_token
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)
    if args.api_key:
        os.environ["MLX_VLM_SERVER_API_KEY"] = args.api_key
    os.environ["MLX_VLM_SERVER_PORT"] = str(args.port)
    if args.log_raw_tokens:
        os.environ["MLX_VLM_LOG_RAW_TOKENS"] = "1"
    if args.seed_request:
        os.environ["MLX_VLM_SEED_REQUEST"] = args.seed_request

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )
