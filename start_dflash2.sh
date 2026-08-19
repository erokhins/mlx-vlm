#!/usr/bin/env bash
set -euo pipefail

# One-command install + serve for the Junie local server with DFlash 2.
#
# Same bootstrap as ./start.sh, but pairs Qwen3.8 with the DFlash 2
# drafter instead of Qwen3.6 + MTP:
#   1) python venv      -> created at ./.venv on first run
#   2) model weights    -> downloaded/verified into ~/.local/share/junie-local
#   3) Junie descriptor -> written to ~/.junie/models
#   4) server           -> mlx_vlm.server on port 8085
#
# Steps 1-3 are no-ops when already done, so this is also the everyday
# start command.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Everything below goes both to the screen and to mlx_server_dflash2.log
# in the repo dir (gitignored; truncated on each start).
exec > >(tee "$SCRIPT_DIR/mlx_server_dflash2.log") 2>&1

PORT=8085
MODEL_ID="mlx-community/Qwen3.8-27B-4bit"
# DFlash 2 block-diffusion drafter for the model above. Auto-detected from
# architectures: DFlash2DraftModel (2-tap dynamic conv + top-k path
# selector). --draft-block-size 5 is the recommended override for 4-bit
# Qwen3.8 targets. Per-request acceptance shows up in the log as
# "Speculative decode: ... accepted_tokens_per_round=".
DRAFT_MODEL_ID="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_BLOCK_SIZE=5

# ---------------------------------------------------------------------------
# 1) Python environment (first run only).
#
# mlx>=0.32 ships wheels for CPython 3.10-3.14 only, while the stock
# /usr/bin/python3 from the Xcode Command Line Tools is still 3.9 -- so the
# venv is built with uv, which downloads a suitable managed CPython on its
# own. uv itself is auto-installed into <repo>/.uv when not already
# present, so the script has no prerequisites at all and everything it
# bootstraps stays inside the repo dir (gitignored).
# ---------------------------------------------------------------------------
VENV="$SCRIPT_DIR/.venv"
VENV_MARKER="$VENV/.deps-installed"
VENV_PYTHON_VERSION=3.13
UV_DIR="$SCRIPT_DIR/.uv"

# Managed-CPython downloads also go inside the repo dir (uv's default is
# ~/.local/share/uv/python).
export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"

find_uv() {
  command -v uv 2>/dev/null && return 0
  for cand in "$UV_DIR/bin/uv" "$HOME/.local/bin/uv"; do
    if [ -x "$cand" ]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

if ! UV_BIN="$(find_uv)"; then
  echo "Installing uv (Python package manager) to $UV_DIR/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_DIR/bin" INSTALLER_NO_MODIFY_PATH=1 sh
  UV_BIN="$UV_DIR/bin/uv"
fi

# A venv built by an older run with Python <3.10 can never install the
# dependencies -- rebuild it.
if [ -x "$VENV/bin/python" ] \
   && ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1; then
  echo "Existing virtualenv uses Python <3.10; recreating it..."
  rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating virtualenv at $VENV (Python $VENV_PYTHON_VERSION via uv) ..."
  venv_setup_failed() {
    rm -rf "$VENV"
  }
  trap venv_setup_failed EXIT
  # Downloads a managed CPython automatically when the machine has none.
  "$UV_BIN" venv --python "$VENV_PYTHON_VERSION" "$VENV"
  "$UV_BIN" pip install --python "$VENV/bin/python" -r "$SCRIPT_DIR/requirements.txt"
  touch "$VENV_MARKER"
  trap - EXIT
elif [ ! -f "$VENV_MARKER" ]; then
  # Venv exists but a previous run died before finishing the dependency
  # install (or predates the marker). The install is a fast no-op when
  # everything is already satisfied.
  echo "Verifying virtualenv dependencies..."
  "$UV_BIN" pip install --python "$VENV/bin/python" -r "$SCRIPT_DIR/requirements.txt"
  touch "$VENV_MARKER"
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# ---------------------------------------------------------------------------
# 2) Model weights (Hugging Face hub cache, same layout start.sh uses).
#    DFlash 2 checkpoints are public, so they are pulled with
#    huggingface_hub instead of the JetBrains zip mirrors.
# ---------------------------------------------------------------------------
BASE_DIR="$HOME/.local/share/junie-local"
MODELS_DIR="$BASE_DIR/models"
export HF_HUB_CACHE="$MODELS_DIR"

hf_cache_dir() {
  echo "$MODELS_DIR/models--${1//\//--}"
}

model_cached() {
  repo_id="$1"
  cache_dir="$(hf_cache_dir "$repo_id")"
  [ -d "$cache_dir/snapshots" ] \
    && [ -n "$(find "$cache_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)" ]
}

download_model_if_needed() {
  repo_id="$1"
  if model_cached "$repo_id"; then
    echo "Model $repo_id is already in $MODELS_DIR"
    return 0
  fi

  echo "Downloading $repo_id into $MODELS_DIR ..."
  mkdir -p "$MODELS_DIR"
  "$PYTHON_BIN" - "$repo_id" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
snapshot_download(repo_id=repo_id)
print(f"  {repo_id} ready.")
PY
}

download_model_if_needed "$MODEL_ID"
download_model_if_needed "$DRAFT_MODEL_ID"

# ---------------------------------------------------------------------------
# 3) Junie model descriptor.
#
# The id must be the real HF repo id (slash form) because Junie sends it
# verbatim as the "model" field and the server loads that repo.
# enable_thinking stays disabled -- the server side relies on it (see
# --preserve-thinking below).
# ---------------------------------------------------------------------------
JUNIE_MODELS_DIR="$HOME/.junie/models"
JUNIE_MODEL_NAME="local-qwen3.8-27b-4bit-dflash2"
JUNIE_MODEL_FILE="$JUNIE_MODELS_DIR/$JUNIE_MODEL_NAME.json"
mkdir -p "$JUNIE_MODELS_DIR"
cat > "$JUNIE_MODEL_FILE" <<EOF
{
  "id": "$MODEL_ID",
  "baseUrl": "http://localhost:$PORT/v1/chat/completions",
  "apiType": "OpenAICompletion",
  "temperature": 0.6,
  "maxContextLength": 150000,
  "extraBody": {
    "enable_thinking": false
  }
}
EOF
echo "Junie model descriptor: $JUNIE_MODEL_FILE"

# Set this model as Junie's default (same mechanism as junie-local's
# install.sh: descriptor-file models are addressed as "custom:<file stem>").
JUNIE_SETTINGS="$HOME/.junie/settings.json"
if [ -f "$JUNIE_SETTINGS" ]; then
  plutil -replace "modelForLaunch" -string "custom:$JUNIE_MODEL_NAME" \
    "$JUNIE_SETTINGS"
  echo "Junie default model set to $JUNIE_MODEL_NAME (restart Junie to apply)."
else
  echo "WARNING: Junie settings not found at $JUNIE_SETTINGS;"
  echo "         select the $MODEL_ID model in Junie manually."
fi

# ---------------------------------------------------------------------------
# 4) Server.
# ---------------------------------------------------------------------------

# The models live in a Hugging Face hub-style cache dir (models--org--name/
# snapshots/...) outside the default HF cache location. Point the HF cache
# at it and load by repo id, offline, so "/v1/models" reports a clean id
# instead of a raw filesystem path.
export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Cross-request KV cache reuse (Automatic Prefix Caching). Verify via
# "APC enabled (...)" and GET /v1/cache/stats.
# Note: --preserve-thinking (below) is required for warm hits to survive new
# user turns -- without it the chat template re-renders older assistant
# turns (drops their <think> blocks) whenever a new user message arrives,
# which changes the token stream mid-history and misses the cache.
export APC_ENABLED=1
export APC_EXACT_SESSIONS=2        # concurrent conversations kept warm
export APC_SESSION_CHECKPOINTS=8   # resumable positions per conversation
# Persist the pinned seed snapshot on SSD so it survives restarts (only the
# seed is written -- APC_DISK_EXACT_SCOPE defaults to "pinned", so the disk
# tier stays at ~1 GB instead of one multi-GB snapshot per request).
export APC_DISK_PATH="$BASE_DIR/apc-cache"

# Stable cross-session prompt prefix (Junie system message + tool schemas +
# first user message; byte-identical across sessions). Prefilled once at
# startup, pinned in APC (never evicted, doesn't count against
# APC_EXACT_SESSIONS), and persisted via APC_DISK_PATH — so the FIRST
# request of a brand-new Junie session already warm-starts. Watch for
# "Seed prefix warmed and pinned" in the log.
SEED_REQUEST="$SCRIPT_DIR/research/junie.json"

# W8A8 int8 prefill on the M5 neural accelerators (see
# research/int8-nax/README.md). int8 weight tensors are built per layer by a
# fused kernel and freed right after use (MLX_VLM_INT8_CACHE=none default),
# so peak memory overhead is ~one layer, not a 24 GB copy; the larger
# prefill step amortizes the per-chunk rebuild (4096 measured best).
# If a quality issue shows up on real workloads, first try
# MLX_VLM_INT8_SCOPE=mlp (keeps attention numerics untouched), then drop
# --int8-prefill entirely.
exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --draft-model "$DRAFT_MODEL_ID" \
  --draft-kind dflash \
  --draft-block-size "$DRAFT_BLOCK_SIZE" \
  --int8-prefill \
  --prefill-step-size 4096 \
  --preserve-thinking \
  --seed-request "$SEED_REQUEST" \
  --log-raw-tokens
