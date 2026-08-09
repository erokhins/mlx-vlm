#!/usr/bin/env bash
set -euo pipefail

# Thin curl wrapper around the local server's HTTP API (see JUNIE_API.md).
#
# Usage:
#   ./serverctl.sh start                    launch the server (background, silent);
#                                           from a checkout, ./init_dev.sh first
#   ./serverctl.sh restart                  gracefully restart the managed server
#   ./serverctl.sh status                   lifecycle phase + inference progress
#   ./serverctl.sh wait                     poll status until phase is "ready"
#   ./serverctl.sh settings                 current serving settings
#   ./serverctl.sh apply key=value [...]    apply settings, restarting the worker
#                                           when the setting requires it:
#                                             ./serverctl.sh apply max_context_length=150000
#                                             ./serverctl.sh apply auto_unload_time=600
#                                             ./serverctl.sh apply kv_quantization=true force=true
#                                           numbers/true/false/null are sent as-is,
#                                           anything else as a JSON string
#   ./serverctl.sh apply-json '{"max_context_length": 150000}'
#   ./serverctl.sh stop                     gracefully stop and unregister it
#   ./serverctl.sh health | models | metrics | cache-stats | unload
#
# HTTP commands drive a checkout and the frozen junie-mlx-vlm alike. Lifecycle
# commands use the current user's launchd domain. PORT overrides the port read
# from the config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The port the daemon serves on, from the same file the daemon reads, with the
# same fallback as DEFAULT_CONFIG["port"] in mlx_vlm_shared/server_settings.py
# for when that file has not been written yet. plutil parses JSON, so no
# interpreter is needed; it fails alike on a missing file, a missing key and
# unparsable contents, and any of those means "use the default".
DEFAULT_PORT=19239
DEFAULT_WORKER_PORT=19240
CONFIG_PATH="${JUNIE_SERVER_CONFIG:-$HOME/.local/share/junie-local/server-config.json}"
PORT="${PORT:-$(plutil -extract port raw -o - -- "$CONFIG_PATH" 2>/dev/null || true)}"
case "$PORT" in
  '' | *[!0-9]*) PORT="$DEFAULT_PORT" ;;
esac
WORKER_PORT="$(plutil -extract worker_port raw -o - -- "$CONFIG_PATH" 2>/dev/null || true)"
case "$WORKER_PORT" in
  '' | *[!0-9]*) WORKER_PORT="$DEFAULT_WORKER_PORT" ;;
esac
BASE="http://localhost:$PORT"
WORKER_BASE="http://localhost:$WORKER_PORT"

# The daemon's own output, beside the worker log it writes itself.
DAEMON_LOG="${CONFIG_PATH%/*}/junie-mlx-vlm-daemon.log"

# One per-user LaunchAgent supervises only the gateway. The gateway remains
# responsible for its inference worker, and the worker's parent watchdog
# stops it when the gateway disappears.
LAUNCHD_LABEL="${JUNIE_LAUNCHD_LABEL:-com.junie.mlx-vlm}"
LAUNCHD_DOMAIN="gui/$(id -u)"
LAUNCHD_SERVICE="$LAUNCHD_DOMAIN/$LAUNCHD_LABEL"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"

usage() {
  sed -n '/^# Usage:/,/^$/{s/^# \{0,1\}//p;}' "${BASH_SOURCE[0]}"
  exit 1
}

CURL=(curl -sS --fail-with-body -m 30)

# plutil reprints JSON but sorts the keys, and refuses anything that is not a
# plist or JSON -- so fall back to the raw body, since an error page is still
# worth reading.
pretty() {
  body="$(cat)"
  if formatted="$(printf '%s' "$body" | plutil -convert json -r -o - -- - 2>/dev/null)"; then
    printf '%s\n' "$formatted"
  else
    printf '%s\n' "$body"
  fi
}

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

find_server() {
  server=""
  for candidate in \
    "$SCRIPT_DIR/junie-mlx-vlm" \
    "$SCRIPT_DIR/.venv/bin/junie-mlx-vlm"; do
    if [ -x "$candidate" ]; then
      server="$candidate"
      break
    fi
  done
  if [ -z "$server" ]; then
    server="$(command -v junie-mlx-vlm || true)"
  fi
  if [ -z "$server" ]; then
    echo "ERROR: no junie-mlx-vlm beside this script, in ./.venv/bin or on" >&2
    echo "       PATH. From a checkout, run ./init_dev.sh first." >&2
    exit 1
  fi
  printf '%s\n' "$server"
}

xml_escape() {
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g'
}

launchd_is_loaded() {
  launchctl print "$LAUNCHD_SERVICE" >/dev/null 2>&1
}

launchd_pid() {
  launchctl print "$LAUNCHD_SERVICE" 2>/dev/null \
    | awk '/^[[:space:]]*pid = / { print $3; exit }'
}

write_launch_agent() {
  server="$1"
  mkdir -p "$(dirname "$LAUNCHD_PLIST")" "$(dirname "$DAEMON_LOG")"
  tmp_plist="$LAUNCHD_PLIST.tmp.$$"
  server_xml="$(xml_escape "$server")"
  config_xml="$(xml_escape "$CONFIG_PATH")"
  log_xml="$(xml_escape "$DAEMON_LOG")"
  label_xml="$(xml_escape "$LAUNCHD_LABEL")"

  cat >"$tmp_plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label_xml</string>
  <key>ProgramArguments</key>
  <array>
    <string>$server_xml</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>JUNIE_SERVER_CONFIG</key>
    <string>$config_xml</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>5</integer>
  <key>StandardOutPath</key>
  <string>$log_xml</string>
  <key>StandardErrorPath</key>
  <string>$log_xml</string>
</dict>
</plist>
EOF

  if ! plutil -lint "$tmp_plist" >/dev/null; then
    rm -f "$tmp_plist"
    echo "ERROR: generated an invalid LaunchAgent plist." >&2
    return 1
  fi
  mv -f "$tmp_plist" "$LAUNCHD_PLIST"
}

prepare_daemon_log() {
  mkdir -p "$(dirname "$DAEMON_LOG")"
  if [ -f "$DAEMON_LOG" ]; then
    mv -f "$DAEMON_LOG" "$DAEMON_LOG.0"
  fi
}

wait_for_server_stop() {
  attempts=0
  while launchd_is_loaded \
    || curl -sS -o /dev/null -m 1 "$BASE/health" >/dev/null 2>&1 \
    || curl -sS -o /dev/null -m 1 "$WORKER_BASE/ready" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge 100 ]; then
      echo "ERROR: server did not stop within 10 seconds." >&2
      return 1
    fi
    sleep 0.1
  done
}

# The one command that cannot be binary agnostic, because it has to know what
# to run. It is the same command either way -- an unpacked tarball has the
# junie-mlx-vlm binary beside this script, a checkout has it in the venv that
# init_dev.sh builds, and it may simply be on PATH.
start_server() {
  server="$(find_server)"

  if launchd_is_loaded; then
    pid="$(launchd_pid)"
    if [ -n "$pid" ]; then
      echo "Already managed by launchd (pid $pid); use ./serverctl.sh wait."
      return 0
    fi
    pid="$(launchctl kickstart -p "$LAUNCHD_SERVICE")"
    echo "Started $LAUNCHD_LABEL through launchd (pid $pid)."
    return 0
  fi

  if curl -sS -o /dev/null -m 2 "$BASE/health" >/dev/null 2>&1; then
    echo "ERROR: port $PORT is served by an unmanaged process." >&2
    echo "       Stop that process before enabling launchd supervision." >&2
    return 1
  fi

  write_launch_agent "$server"
  prepare_daemon_log
  launchctl bootstrap "$LAUNCHD_DOMAIN" "$LAUNCHD_PLIST"
  echo "Started $LAUNCHD_LABEL through launchd; logging to $DAEMON_LOG"
  echo "Follow it with ./serverctl.sh wait"
}

stop_server() {
  if launchd_is_loaded; then
    launchctl bootout "$LAUNCHD_SERVICE"
    rm -f "$LAUNCHD_PLIST"
    wait_for_server_stop
    echo "Stopped $LAUNCHD_LABEL and removed its LaunchAgent."
    return 0
  fi

  # Backward-compatible cleanup for a daemon started by the old nohup path.
  rm -f "$LAUNCHD_PLIST"
  if curl -sS -o /dev/null -m 2 "$BASE/health" >/dev/null 2>&1; then
    post /shutdown
    wait_for_server_stop
  else
    echo "Server is already stopped."
  fi
}

restart_server() {
  stop_server
  start_server
}

wait_ready() {
  while :; do
    phase="$("${CURL[@]}" -m 5 "$BASE/status" 2>/dev/null \
      | plutil -extract phase raw -o - -- - 2>/dev/null || true)"
    echo "phase: ${phase:-unreachable}"
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
  start) start_server ;;
  restart) restart_server ;;
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
  stop) stop_server ;;
  health) get /health ;;
  models) get /v1/models ;;
  metrics) get /metrics ;;
  cache-stats) get /v1/cache/stats ;;
  unload) post /unload ;;
  *) usage ;;
esac
