# Junie local server — install & run (from sources)

Serves `mlx-community/Qwen3.6-27B-4bit` on Apple Silicon as an
OpenAI-compatible endpoint for Junie, with MTP + n-gram speculative
decoding, prefix caching (APC), and int8 NAX prefill.

## 1. Clone

```bash
git clone -b local/serving-patches https://github.com/erokhins/mlx-vlm
cd mlx-vlm
```

## 2. Prerequisites

Installed separately (by the packaged installer):

- model weights in the configured `models_dir`
  (`~/.local/share/junie-local/models/` by default), in HF-hub layout
  (`models--mlx-community--Qwen3.6-27B-4bit` and `...-MTP-4bit`, ~17 GB) —
  the worker loads them from there by repo id, offline;
- `~/.junie/models/local-qwen3.6-27b-4bit-vlm.json`, the Junie model
  descriptor pointing at `http://localhost:19239/v1/chat/completions`, and
  `modelForLaunch` in `~/.junie/settings.json` set to
  `custom:local-qwen3.6-27b-4bit-vlm`.

Restart Junie after either changes.

## 3. Start

```bash
./init_dev.sh          # once, and again when uv.lock changes
./serverctl.sh start
```

`init_dev.sh` only creates the Python virtualenv and installs dependencies
(via `uv`, which is itself auto-installed and downloads Python 3.13 if the
machine has none — the stock macOS `python3` is too old for `mlx>=0.32`).
The project is installed editable, so the `junie-mlx-vlm` it puts in
`./.venv/bin` runs this checkout's sources.

`serverctl.sh start` is the entrypoint in both worlds: it runs that
`junie-mlx-vlm` from a checkout, and the frozen one from
`build_cli_tarball.sh` beside it on a shipped machine. It installs a per-user
macOS LaunchAgent and returns immediately. `launchd` restarts the gateway if
it crashes; the gateway still owns and supervises the inference worker.

The daemon reads every setting from `server-config.json`, serves the public
API on its `host`/`port` (`0.0.0.0:19239` by default), and spawns the
inference worker itself on `worker_port` (`19240`).

Startup seeding — prefilling and pinning the shared Junie prompt prefix so
new sessions warm-start — is **currently disabled** pending rework. Point
`seed_request` at a chat-completions request body to turn it back on for one
machine; nothing does so by default.

Stop it with `./serverctl.sh stop`. This unregisters the LaunchAgent, stops
the gateway and worker gracefully, and releases the model memory. Because the
plist is removed, a manual stop stays stopped after the next login.

Model weights and the Junie model descriptor are **not** installed by this
script — see [Prerequisites](#2-prerequisites).

## 4. Benchmark (optional)

With the server running:

```bash
./bench.sh
```

Replays a captured real Junie session (11 requests) and prints per-request
serving stats plus their means:

```
00.json: cached=18506/18522 (100%) | prefill_new=16tok @81tok/s | decode=66tok @45.5tok/s | accept=65% (2.44tok/round) ngram=3r/3tok
...
mean over 11 requests: KV cached 100% (...) | prefill 70tok/s over 176 new tokens | generation 37.1tok/s (...) | accept 52% of drafted (2.51tok/round) | ngram share 24% of output
```

(A first run on a fresh cache shows real cold-prefill numbers; repeat runs
are fully KV-cached.)

## Paths used / generated

| Path | What |
|---|---|
| `<repo>/.venv/` | Python virtualenv (created on first run) |
| `~/Library/LaunchAgents/com.junie.mlx-vlm.plist` | per-user launchd service created by `serverctl.sh start` and removed by `serverctl.sh stop` |
| `~/.local/share/junie-local/junie-mlx-vlm-daemon.log` | the daemon's own output; previous run kept as `.log.0` |
| `~/.local/share/junie-local/junie-mlx-vlm.log` | the inference worker's output, appended across restarts within a run; previous run kept as `.log.0` |
| `~/.local/share/junie-local/server-config.json` | the only config: model, models dir, host/port, worker port, context, KV quantization, idle timeout, and worker launch settings (override its location with `JUNIE_SERVER_CONFIG`) |
| `<repo>/research/junie.json` | the stable Junie prompt prefix (system message + tool schemas + first user message); used by `bench.sh`, and the request body `seed_request` expects — not loaded at startup while seeding is disabled |
| `<repo>/research/junie-replay/` | captured session + replay script used by `bench.sh` |
| `~/.local/share/junie-local/models/` | model weights, HF-hub layout (`models--mlx-community--Qwen3.6-27B-4bit`, `...-MTP-4bit`); installed separately, relocatable via the `models_dir` setting |
| `~/.local/share/junie-local/apc-cache/` | APC disk tier — holds pinned snapshots (~1 GB each) so they survive restarts; empty while seeding is disabled |
| `<repo>/.uv/bin/uv` | `uv` binary (only when not already installed on the machine) |
| `<repo>/.uv/python/` | uv-managed CPython 3.13 (only when the machine has no suitable Python) |
| `~/.junie/models/local-qwen3.6-27b-4bit-vlm.json` | Junie model descriptor pointing at this server; installed separately |
| `~/.junie/settings.json` | existing Junie settings; `modelForLaunch` points at this model |

## Server endpoints

| URL | What |
|---|---|
| `http://localhost:19239/v1/chat/completions` | OpenAI-compatible chat endpoint (this is what Junie calls); responses include a `timings` block with prefill/decode speeds and speculative-acceptance counters |
| `http://localhost:19239/status` | lifecycle, loaded-model state, and active-request count |
| `http://localhost:19239/current_settings` | persistent Junie serving settings |
| `POST http://localhost:19239/apply_settings` | apply settings; restart the worker when required |
| `POST http://localhost:19239/shutdown` | stop the worker and gateway and release model memory |
| `http://localhost:19239/health` | cheap gateway liveness, even while the worker is stopped |
| `http://localhost:19239/metrics` | inference metrics plus gateway process state |
| `http://localhost:19239/cache/stats` | prompt/KV cache statistics |

The gateway stays available if MLX or the worker process crashes. It returns
`503` for the interrupted request and starts a fresh worker. A manual
settings restart is rejected with `409` while inference is active unless the
request contains `"force": true`. If the gateway process itself crashes,
`launchd` starts a new gateway; the old worker's parent watchdog stops that
worker, and the new gateway starts a replacement.

Use the control script instead of hand-written curl commands:

```bash
./serverctl.sh status
./serverctl.sh settings
./serverctl.sh apply auto_unload_time=600
./serverctl.sh apply max_context_length=150000
./serverctl.sh wait
./serverctl.sh restart
./serverctl.sh stop
```

Changing only `auto_unload_time` is live. Model, context-limit, and KV-cache
changes stop the worker, atomically save the config, and launch a new worker.
If inference is active, the request is rejected with `409` unless you add
`force=true`.
See [JUNIE_API.md](JUNIE_API.md) for exact request and response formats.

When the configured idle timeout expires, the gateway kills the worker and
releases model memory. The next chat request starts a new worker, waits for the
model to become ready, and then forwards that original request. The daemon
is the only supported process entrypoint; never start a worker beside it.

Chat requests are batch-only; an incoming `stream: true` is changed to
`false`. At `soft_request_timeout` seconds (270 by default) the worker cancels
only that generation and returns the partial answer when one exists, or `504`
when it produced nothing. If the worker cannot stop it by the daemon's
275-second hard limit, the daemon restarts the worker. Two consecutive worker `500` responses also trigger a restart;
client errors such as `400` and `422` do not.

## What to expect in the log

- `Seed prefix warmed and pinned: ... cached_tokens=14551 elapsed=0.5s` —
  the cross-session prefix is ready; new Junie sessions warm-start. Only
  appears when `seed_request` is set, which it is not by default.
- `Prefill completed: ... cached_tokens=...` — APC hit size per request.
- `Speculative decode: ... accepted_tokens_per_round=... ngram_rounds=...`
  — MTP drafter + n-gram prompt-lookup acceptance per request.
- `Raw generated tokens: ...` — each response's tokens as text, with
  accepted MTP drafts in green and accepted n-gram drafts in cyan
  (`--log-raw-tokens`; view with `tail -f` or `less -R`).

## Tuning knobs (already set to measured optima)

All optional; see `research/mtp-overhead/README.md` for the measurements
behind the defaults:

- `MLX_VLM_NGRAM_*` — n-gram prompt-lookup drafting (base window 4,
  full-accept doubling to 32; `MLX_VLM_NGRAM_DRAFT=0` disables).
- `APC_EXACT_SESSIONS` / `APC_SESSION_CHECKPOINTS` — warm-conversation
  capacity.
- `APC_DISK_EXACT_SCOPE=all` — persist every conversation snapshot to disk
  (default `pinned` keeps only pinned ones, of which there are none while
  seeding is disabled).
- `MLX_VLM_INT8_SCOPE=mlp` or dropping `--int8-prefill` — fallback if
  prefill quality issues ever show up.
