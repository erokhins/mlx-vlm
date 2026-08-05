#!/usr/bin/env bash
set -euo pipefail

# Thin curl wrapper around the local server's HTTP API (see JUNIE_API.md).
#
# Usage:
#   ./serverctl.sh start                    launch the server (./start.sh, background)
#   ./serverctl.sh status                   lifecycle phase + inference progress
#   ./serverctl.sh wait                     poll status until phase is "ready"
#   ./serverctl.sh settings                 current serving settings
#   ./serverctl.sh apply key=value [...]    apply settings (restarts model serving,
#                                           except auto_unload_time which applies live):
#                                             ./serverctl.sh apply max_context_length=150000
#                                             ./serverctl.sh apply kv_quantization=true
#                                             ./serverctl.sh apply auto_unload_time=600
#                                             ./serverctl.sh apply max_context_length=null force=true
#                                           numbers/true/false/null are sent as-is,
#                                           anything else as a JSON string
#   ./serverctl.sh apply-json '{"max_context_length": 150000}'
#   ./serverctl.sh stop                     POST /shutdown (graceful)
#   ./serverctl.sh health | models | metrics | cache-stats | cache-reset | unload
#
# The port comes from server-config.json (PORT env overrides); API_KEY
# adds a bearer token.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUNIE_SERVER_CONFIG="${JUNIE_SERVER_CONFIG:-$HOME/.local/share/junie-local/server-config.json}"
if [ -z "${PORT:-}" ]; then
  PORT=$(sed -n 's/^[[:space:]]*"port"[^0-9]*\([0-9][0-9]*\).*/\1/p' \
    "$JUNIE_SERVER_CONFIG" 2>/dev/null | head -1)
fi
PORT="${PORT:-19239}"
BASE="http://localhost:$PORT"

usage() {
  sed -n '/^# Usage:/,/^$/{s/^# \{0,1\}//p;}' "${BASH_SOURCE[0]}"
  exit 1
}

if command -v python3 >/dev/null 2>&1; then
  pretty() { python3 -m json.tool; }
else
  pretty() { cat; }
fi

CURL=(curl -sS --fail-with-body -m 30)
if [ -n "${API_KEY:-}" ]; then
  CURL+=(-H "Authorization: Bearer $API_KEY")
fi

get() { "${CURL[@]}" "$BASE$1" | pretty; }
post() {
  if [ $# -ge 2 ]; then
    "${CURL[@]}" -X POST -H 'Content-Type: application/json' -d "$2" "$BASE$1" | pretty
  else
    "${CURL[@]}" -X POST "$BASE$1" | pretty
  fi
}

kv_to_json() {
  json="{"
  sep=""
  for pair in "$@"; do
    case "$pair" in
      *=*) ;;
      *) echo "ERROR: expected key=value, got '$pair'" >&2; exit 1 ;;
    esac
    key="${pair%%=*}"
    value="${pair#*=}"
    case "$value" in
      true | false | null) ;;
      *)
        if ! [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
          value="\"$value\""
        fi
        ;;
    esac
    json="$json$sep\"$key\": $value"
    sep=", "
  done
  echo "$json}"
}

wait_ready() {
  while :; do
    phase=$("${CURL[@]}" -m 5 "$BASE/status" 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])' \
      2>/dev/null || echo unreachable)
    echo "phase: $phase"
    case "$phase" in
      ready) return 0 ;;
      error)
        get /status
        return 1
        ;;
    esac
    sleep 2
  done
}

cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
  start) exec "$SCRIPT_DIR/start.sh" ;;
  status) get /status ;;
  wait) wait_ready ;;
  settings) get /current_settings ;;
  apply)
    [ $# -gt 0 ] || usage
    post /apply_settings "$(kv_to_json "$@")"
    ;;
  apply-json)
    [ $# -eq 1 ] || usage
    post /apply_settings "$1"
    ;;
  stop) post /shutdown ;;
  health) get /health ;;
  models) get /v1/models ;;
  metrics) get /metrics ;;
  cache-stats) get /v1/cache/stats ;;
  cache-reset) post /v1/cache/reset ;;
  unload) post /unload ;;
  *) usage ;;
esac
