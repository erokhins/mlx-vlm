#!/usr/bin/env bash
set -euo pipefail

# One-command install + serve for the Junie local server.
#
# On every run this script makes sure the pieces are in place, then starts
# the OpenAI-compatible server from this repo's sources:
#   1) model weights   -> downloaded/verified into ~/.local/share/junie-local
#   2) Junie descriptor -> written to ~/.junie/models
#   3) python venv      -> created at ./.venv on first run
#   4) server           -> mlx_vlm.server (port from server-config.json,
#                          default 8085)
#
# Steps 1-3 are no-ops when already done, so this is also the everyday
# start command.
#
# The script is non-blocking and silent: it relaunches itself in the
# background and returns immediately; all output goes to mlx_server.log.
# Watch startup with GET /status (phase: loading_model -> warming_up ->
# ready) and stop the server with POST /shutdown.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="$HOME/.local/share/junie-local"

# Persistent server settings: model/drafter, the runtime settings the
# /apply_settings API manages, and the launch settings (host/port, prefill
# tuning, seed request, ...). The server reads it via
# `python -m mlx_vlm.server.junie` and creates it with defaults on first
# start; edit it by hand while the server is stopped.
export JUNIE_SERVER_CONFIG="$BASE_DIR/server-config.json"

# The port lives in the config file; fall back to the default until the
# first start creates it.
PORT=$(sed -n 's/^[[:space:]]*"port"[^0-9]*\([0-9][0-9]*\).*/\1/p' \
  "$JUNIE_SERVER_CONFIG" 2>/dev/null | head -1)
PORT=${PORT:-8085}

LOG_FILE="$SCRIPT_DIR/mlx_server.log"

if [ "${1:-}" != "--foreground" ]; then
  # A server already answering on the port stays as it is.
  if curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    exit 0
  fi
  nohup "$SCRIPT_DIR/start.sh" --foreground >/dev/null 2>&1 </dev/null &
  exit 0
fi

# Everything below goes to mlx_server.log in the repo dir (gitignored;
# truncated on each start).
exec >"$LOG_FILE" 2>&1

# These ids drive the weight download and the Junie descriptor below. The
# model the server actually serves (and its MTP speculative-decoding
# drafter) is decided by the persistent config file (JUNIE_SERVER_CONFIG,
# see section 4), whose defaults are these same ids — keep them in sync
# with DEFAULT_CONFIG in mlx_vlm/server/junie/config.py.
#
# Drafter notes: it has no standalone language_model head, so it is only
# ever a draft model, never requested directly as a chat "model".
# Measured ~1.45-1.6x decode speedup at 2.24 accepted tokens/round; the
# drafter itself costs only ~9% of a round — the rest is the 3-token verify
# forward. The configured draft depth 3 is optimal (research/mtp-overhead:
# 2/4/5/6 are all slower) and the adaptive controller already handles
# bursts. Per-request acceptance shows up in the log as
# "Speculative decode: ... accepted_tokens_per_round=".
MODEL_ID="mlx-community/Qwen3.6-27B-4bit"
DRAFT_MODEL_ID="mlx-community/Qwen3.6-27B-MTP-4bit"

# ---------------------------------------------------------------------------
# 1) Model weights (download style borrowed from junie-local's install.sh:
#    resumable curl with retry/backoff, SHA256 verification, HF-hub-layout
#    zips extracted with completion markers so interrupted installs redo
#    cleanly).
# ---------------------------------------------------------------------------
BASE_URL="https://download.jetbrains.com/resources/junie-local"
MODELS_DIR="$BASE_DIR/models"
DOWNLOAD_DIR="$BASE_DIR/incomplete_downloads"

MODEL_ZIP_1="models--mlx-community--Qwen3.6-27B-4bit.zip"
MODEL_SHA256_1="adf7f8d832ed994dcc6d09372036b4d12f49a4ccda066179cc64dc2dd113f91d"
MODEL_DIR_ID_1="mlx-community--Qwen3.6-27B-4bit"
MODEL_ZIP_2="models--mlx-community--Qwen3.6-27B-MTP-4bit.zip"
MODEL_SHA256_2="9266c1ba244ec6176fc82474bbfd20614969eb28c4cfa24301e515fbd1f5a525"
MODEL_DIR_ID_2="mlx-community--Qwen3.6-27B-MTP-4bit"

download_with_retry() {
  url="$1"
  output_file="$2"
  max_retries="${3:-3}"
  attempt=1
  delay=2

  while [ "$attempt" -le "$max_retries" ]; do
    echo "  Attempt $attempt of $max_retries..."
    if curl -sSL -C - -o "$output_file" "$url"; then
      return 0
    fi

    if [ "$attempt" -lt "$max_retries" ]; then
      echo "  Download failed. Retrying in ${delay}s..."
      sleep "$delay"
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done

  echo "  ERROR: Download failed after $max_retries attempts."
  return 1
}

download_and_verify() {
  archive="$1"
  expected_sha256="$2"

  echo "Downloading $archive..."
  download_with_retry "$BASE_URL/$archive" "$DOWNLOAD_DIR/$archive"
  echo "  Download complete. Checking SHA256..."

  actual=$(shasum -a 256 "$DOWNLOAD_DIR/$archive" | awk '{print $1}')
  if [ "$actual" != "$expected_sha256" ]; then
    echo "  ERROR: SHA256 mismatch for $archive"
    echo "    Expected: $expected_sha256"
    echo "    Actual:   $actual"
    exit 1
  fi
  echo "  SHA256 verified: $actual"
}

model_completion_marker() {
  echo "$MODELS_DIR/.models--$1.installed"
}

model_installed() {
  model_dir_id="$1"
  [ -d "$MODELS_DIR/models--$model_dir_id" ] \
    && [ -f "$(model_completion_marker "$model_dir_id")" ]
}

install_model_if_needed() {
  zip_file="$1"
  sha256_sum="$2"
  model_dir_id="$3"

  if model_installed "$model_dir_id"; then
    return 0
  fi

  echo "Model $model_dir_id is not installed. Downloading..."
  mkdir -p "$MODELS_DIR" "$DOWNLOAD_DIR"
  download_and_verify "$zip_file" "$sha256_sum"
  echo "Extracting $zip_file to $MODELS_DIR..."
  # Remove leftovers from a previously interrupted extraction
  rm -rf "$MODELS_DIR/models--$model_dir_id"
  unzip -q "$DOWNLOAD_DIR/$zip_file" -d "$MODELS_DIR"
  touch "$(model_completion_marker "$model_dir_id")"
  rm -f "$DOWNLOAD_DIR/$zip_file"
  echo "  Extraction complete."
}

install_model_if_needed "$MODEL_ZIP_1" "$MODEL_SHA256_1" "$MODEL_DIR_ID_1"
install_model_if_needed "$MODEL_ZIP_2" "$MODEL_SHA256_2" "$MODEL_DIR_ID_2"
rmdir "$DOWNLOAD_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2) Junie model descriptor.
#
# The id must be the real HF repo id (slash form) because Junie sends it
# verbatim as the "model" field and the server loads that repo.
# enable_thinking stays disabled -- the server side relies on it (see
# --preserve-thinking below).
# ---------------------------------------------------------------------------
JUNIE_MODELS_DIR="$HOME/.junie/models"
JUNIE_MODEL_NAME="local-qwen3.6-27b-4bit-vlm"
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
# 3) Python environment (first run only).
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

# ---------------------------------------------------------------------------
# 4) Server.
# ---------------------------------------------------------------------------

# The models live in a Hugging Face hub-style cache dir (models--org--name/
# snapshots/...) outside the default HF cache location. Point the HF cache
# at it and load by repo id, offline, so "/v1/models" reports a clean id
# instead of a raw filesystem path.
export HF_HUB_CACHE="$MODELS_DIR"
export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# No serving flags or inference env vars here: everything (host/port,
# prefill tuning [int8_prefill, prefill_step_size], preserve_thinking,
# seed_request, log_raw_tokens, APC [apc_enabled, apc_exact_sessions,
# apc_session_checkpoints, apc_disk_path], ngram_max, model/drafter, KV
# quantization, ...) lives in the config file (JUNIE_SERVER_CONFIG above).
# The junie launcher reads it, exports the inference env vars and builds
# the mlx_vlm.server command line; see DEFAULT_CONFIG in
# mlx_vlm/server/junie/config.py for per-field docs, and
# research/int8-nax/README.md for the int8-prefill background. seed_request
# defaults to research/junie.json in this repo — the stable Junie prompt
# prefix, prefilled and pinned at startup ("Seed prefix warmed and pinned"
# in the log) so the first request of a brand-new session warm-starts.
exec "$PYTHON_BIN" -m mlx_vlm.server.junie
