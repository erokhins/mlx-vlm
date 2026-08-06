#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running server by replaying a captured real Junie session
# (research/junie-replay). Prints per-request serving stats — KV cached,
# prefill speed over new tokens, generation speed, speculative acceptance —
# and their means. Stats come from the response "timings" blocks, so this
# works however the server was started.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The port lives in server-config.json (see start.sh); PORT env overrides.
JUNIE_SERVER_CONFIG="${JUNIE_SERVER_CONFIG:-$HOME/.local/share/junie-local/server-config.json}"
if [ -z "${PORT:-}" ]; then
  # `|| true`: a missing config file fails the pipeline, and under
  # `set -euo pipefail` that would abort silently.
  PORT=$(sed -n 's/^[[:space:]]*"port"[^0-9]*\([0-9][0-9]*\).*/\1/p' \
    "$JUNIE_SERVER_CONFIG" 2>/dev/null | head -1 || true)
fi
PORT=${PORT:-19239}

if ! curl -sf -m 5 "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "Server is not running on port $PORT."
  echo "Start it first:  ./start.sh"
  exit 1
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/research/junie-replay/replay.py" \
  --url "http://localhost:$PORT" "$@"
