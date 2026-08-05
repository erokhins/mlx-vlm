# Change catalog: everything after `3e1ca5e` (branch `erokhins/mac-app`)

Base: `3e1ca5e` — "start.sh: bootstrap Python via repo-local uv".
18 commits, +2650/−154 across 26 files (2026-08-05 … 2026-08-06).

Purpose of this document: a shopping list for a clean rebuild. Each entry
describes the **end state** (intermediate designs that were superseded are
collapsed; supersessions are called out explicitly), why it was done, and
what it depends on. Groups are ordered roughly by how safe/valuable they
are to keep.

Quick orientation — where changes live:

- **Isolated (junie layer + scripts + docs)**: `mlx_vlm/server/junie/*`,
  `start.sh`, `serverctl.sh`, `bench.sh`, `INSTALL.md`, `JUNIE_API.md`.
  Cannot destabilize core serving by themselves.
- **Core server**: `mlx_vlm/server/app.py`, `generation.py`, `openai.py`,
  `runtime.py`, `__init__.py`.
- **Core model/cache** (highest risk): `mlx_vlm/models/cache.py`,
  `mlx_vlm/models/qwen3_5/language.py`, `mlx_vlm/apc.py`,
  `mlx_vlm/apc_adapters.py`.

Commit → group map:

| Commit | Title | Group | Core-touching |
|---|---|---|---|
| `4f1af2d` | REST control plane; non-blocking start.sh | A, C | app.py (hooks), generation.py (progress) |
| `ee2bcbb` | serverctl.sh | C | no |
| `842a690` | 4-field settings API; idle auto-unload | A | no (junie only) |
| `f7ae39d` | unload memory-leak fix | B | app.py (6 lines) |
| `4019ab1` | /status memory stats; cache-reset frees memory | B | app.py (5 lines) |
| `8914657` | server-config.json persistence | D | no |
| `2714def` | kv-quant MTP verify crash fix | E | cache.py, qwen3_5 |
| `444dc9a` | quantized APC session storage | F | apc.py, apc_adapters.py, cache.py |
| `8ced71d` | peak memory: causal verify, step 1024, ngram cap | E/F | qwen3_5 |
| `03fd26c` | all launch flags + inference env → config | D | no |
| `5fa7b2d` | localhost-only default | D | no |
| `4c117ad` | port 19239 | C/D | no |
| `dde81a8` | /shutdown cancels in-flight inference | G | no (junie only) |
| `48eee51` | cancel generation on client disconnect | B | app.py, openai.py |
| `0a35969` | serialize request processing | G | generation.py |
| `86099f4` | quantized row caches in row-split prefill | E | cache.py, qwen3_5 |
| `b6b9602` | token-id-0 bandage | G | generation.py, openai.py, runtime.py |
| `cadcb2e` | guarded reload after auto-unload | G | app.py (hook), runtime.py |

---

## Group A — REST control plane (the Mac-app API)

### A1. `mlx_vlm/server/junie/` package: lifecycle + control endpoints (`4f1af2d`, refined by `842a690`)

**Why:** the Mac app needs to manage the local server over plain HTTP —
observe state, change settings, stop it — without a separate control
process. Design decision (after discussing CLI/pidfile alternatives): the
HTTP server itself is the single control plane.

**End state:**

- `junie/lifecycle.py` — thread-safe phase machine:
  `starting → loading_model → warming_up → ready`, plus `restarting`,
  `stopping`, `error`. Tracks the loader thread id (exempt from guards).
- `junie/control.py` — registered via the repo's existing
  `register_routes(app, deps)` pattern (deps = late-bound lambdas from
  app.py so test monkeypatching keeps working). Endpoints (all also under
  `/v1/`, honoring the optional `--api-key`):
  - `GET /status` — phase + model info + `memory` (see B2) + live
    inference progress: per-request `stage` (queued/prefill/decode),
    `prefill_progress` 0–1, `generated_tokens`, `elapsed_s`, plus
    `in_flight` / `queue_depth`.
  - `GET /current_settings` / `POST /apply_settings` — settings surface
    deliberately reduced (by `842a690`) to exactly four fields:
    `model_name`, `max_context_length` (→ `MAX_KV_SIZE`),
    `kv_quantization` (plain on/off; 8-bit uniform underneath),
    `auto_unload_time`. `apply_settings` restarts **model serving only**
    (never the HTTP server/port) on a background worker guarded by a
    reload lock; `auto_unload_time` alone applies live without restart.
    409 when busy or when inference is in flight (`force: true`
    overrides). Applied settings persist to the config file (D1).
  - `POST /shutdown` — see G2.
- **Model preload moved out of the FastAPI lifespan onto a background
  thread** (small hook in app.py lifespan): HTTP (and `/status`) is
  reachable ~2 s after process start instead of after the multi-minute
  weight load; while a load/reload runs, `get_cached_model` rejects
  inference with a clean 503 (busy-phase guard — ~10 lines in core
  app.py). This also distinguishes `warming_up` from `ready`, which
  `/health` cannot (it returns 200 before the seed warmup finishes).
- The seed-warmup function in app.py gained an optional `on_done`
  callback + bool return (body unchanged) so the junie layer can flip
  `warming_up → ready`.
- `generation.py`: the batching engine's per-request progress dict
  (previously log-only) is exposed on the instance (`_active_requests`)
  and snapshotted read-only by `/status` (the snapshot function lives in
  the junie layer). Core diff: ~11 lines (field init, `prompt_tokens`
  recorded, dict bound in `_run`, post-loop `clear()`).

**Docs:** `JUNIE_API.md` (created — request/response examples for every
endpoint) + `INSTALL.md` updates.

**Pick notes:** foundation for almost everything else (D, G, parts of B).
The core hooks are small and behavior-neutral when the junie layer is
absent.

### A2. Idle auto-unload watchdog (`842a690`, split into `junie/watchdog.py` + `junie/state.py`)

**Why:** free ~20 GB when the model sits idle (Mac-app requirement).

**End state:** background thread, 10 s tick; when `auto_unload_time` is
set and there was no inference activity for that long (idle anchor = max
of model-load time and last completed request; in-flight requests always
block it), it unloads the model. Takes the same reload lock as
`apply_settings` and re-checks under it, so it can never fire mid-restart
or mid-request. Server stays `ready` with `model.loaded: false` in
`/status`. `junie/state.py` holds the shared reload lock + serving-config
dict + in-flight helper (avoids circular imports).

**⚠️ Known consequence:** what happens on the *next* request after an
unload evolved — see G5 and the "Open problem" section. The watchdog is
also the main trigger of the in-process-reload corruption.

---

## Group B — Memory correctness fixes (core, small, high value)

### B1. Unload left ~17 GB in the MLX buffer cache (`f7ae39d`) — genuine core bug

**Why:** `POST /unload` (and the watchdog) "freed" the model but process
footprint barely moved.

**Root cause:** in `unload_model_sync`,
`for cache_group, _ in list(registry.items())` kept every cache dict —
and the weights — referenced until the function returned, so the weights
died only *after* the final `mx.clear_cache()` and their buffers parked
in the MLX buffer cache forever.

**End state (app.py, 6 lines):** iterate group *names* only;
`mx.synchronize()` before the trailing `mx.clear_cache()` (buffer release
rides the Metal stream). Measured: RSS 16.8 GB → 0.6 GB, physical
footprint 21.1 GB → 431 MB after unload. **Keep this regardless of
everything else — it fixes upstream `/unload` too.**

### B2. `/status` memory stats + cache-reset actually frees (`4019ab1`)

**Why:** the app needs "how much memory is the server using" and "which
part is reducible".

**End state:**

- `junie/memory.py`: `memory` in `/status` =
  `{total_gb, peak_gb, kv_cache_gb}`. total/peak are the **process
  physical footprint** via `proc_pid_rusage` (ctypes; validated
  byte-exact against `vmmap`) — RSS undercounts Metal buffers.
  `kv_cache_gb` = in-RAM APC KV (session `kv_bytes` + block pool) — the
  one component freeable without unloading (an earlier
  `active/cache/peak` MLX-counter shape was replaced before commit; only
  this shape ever landed).
- Core fix (app.py, 5 lines): `POST /v1/cache/reset` had the same
  disease as B1 — freed KV parked in the MLX buffer cache. Same
  `gc + synchronize + clear_cache` tail; measured 19.1 → 17.7 GB live.

### B3. Cancel generation when the client disconnects (`48eee51`) — two core bugs

**Why:** a vanished client left inference running to completion (32 k
tokens for nobody).

**End state:**

- **app.py:** the Server-header middleware was `@app.middleware("http")`
  (BaseHTTPMiddleware), which proxies the receive channel and **never
  forwards `http.disconnect`** — `Request.is_disconnected()` stayed False
  forever server-wide, and streaming cancellation only worked lazily via
  SSE write failures. Rewritten as pure ASGI middleware wrapping only
  `send`. **This fix benefits every endpoint; keep it.**
- **openai.py (non-stream chat/completions):** the blocking generate is
  now supervised with a 0.5 s `is_disconnected()` poll; on disconnect the
  token stream's `close()` cancels the batched generation (same per-uid
  mechanism streaming uses), metrics record `client_disconnected`, 499
  returned. Measured: cancellation ~0.5 s after client kill (was: never).
  Note: the Anthropic `/v1/messages` non-stream path was *not* given the
  explicit watcher (only benefits from the middleware fix).

---

## Group C — Deployment scripts

### C1. start.sh: non-blocking, silent, dumb (`4f1af2d`, `8914657`, `03fd26c`, `4c117ad`)

**End state:** start.sh relaunches itself in the background and returns
immediately; all output → `mlx_server.log` (truncated per start); no-op
if the port already answers; `--foreground` for debugging. It keeps only
repo/weights glue (downloads, HF_HUB_CACHE, PYTHONPATH, venv) and ends in
`exec python -m mlx_vlm.server.junie` — **no serving flags at all** (see
D1/D2). It reads the port back out of the config file (sed, fallback
19239) for the already-running check and the Junie descriptor.
Model-download curl switched to `-sS` (no progress bars into the log).

### C2. serverctl.sh (`ee2bcbb`, `4c117ad`)

**End state:** thin curl wrapper: `start`, `status`, `wait` (poll until
ready), `settings`, `apply key=value…` (builds JSON; literals for
numbers/bools/null), `apply-json`, `stop`, `health`, `models`, `metrics`,
`cache-stats`, `cache-reset`, `unload`. `--fail-with-body` so server
error bodies print AND exit codes propagate. Reads the port from the
config (PORT env overrides); optional API_KEY bearer. bench.sh reads the
port the same way.

---

## Group D — Configuration system

### D1. `server-config.json` — persistent, single source of truth (`8914657`, `03fd26c`)

**Why:** settings were lost on every restart, and serving flags were
scattered across start.sh.

**End state:** `~/.local/share/junie-local/server-config.json` (path via
`JUNIE_SERVER_CONFIG` env; feature entirely inert when unset, so bare
`python -m mlx_vlm.server` and tests keep stock behavior).

- Created with defaults on first start; corrupt file → kept aside as
  `.invalid`, recreated; invalid values dropped individually; **upgraded
  in place** when new fields are introduced (missing keys written back).
- `apply_settings` persists the applied subset — only **after** it took
  effect, so a failed reload can't poison the next restart.
- Fields (per-field docs live as comments on `DEFAULT_CONFIG` in
  `junie/config.py`):
  - runtime/API: `model_name`, `draft_model`, `draft_kind` (hand-edit
    only), `max_context_length`, `kv_quantization`, `auto_unload_time`
  - launch: `host` (**default 127.0.0.1** — `5fa7b2d`: no auth without
    --api-key, don't expose to LAN), `port` (**default 19239** —
    `4c117ad`: collision-safe), `int8_prefill` (true),
    `prefill_step_size` (1024 — see E3), `preserve_thinking` (true),
    `seed_request` (null = repo's research/junie.json; "" disables),
    `log_raw_tokens` (true)
  - inference env: `apc_enabled` (true), `apc_exact_sessions` (2),
    `apc_session_checkpoints` (8), `apc_disk_path` (null = apc-cache next
    to the config), `ngram_max` (8 — see E3),
    `max_concurrent_requests` (1 — see G1)

### D2. `python -m mlx_vlm.server.junie` launcher (`03fd26c`)

**Why:** host/port must exist before uvicorn binds — earlier than the
lifespan config read.

**End state:** `junie/launch.py` + `__main__.py`: reads the config,
exports the inference env vars (APC_*, MLX_VLM_NGRAM_MAX,
MLX_VLM_MAX_CONCURRENT_REQUESTS), builds the equivalent `mlx_vlm.server`
argv from the launch fields, and delegates to the stock CLI. Extra CLI
args append after config-derived ones (argparse last-wins) for ad-hoc
overrides. No core changes.

---

## Group E — KV-quantization correctness (core model/cache; only needed if `kv_quantization` is used)

### E1. Quantized KV + MTP verify crashed (`2714def`, verify part superseded by `8ced71d`)

**Why:** with `kv_quantization: true`, any long-context request (past
`QUANTIZED_KV_START`=5000) 500'd: the MTP target-verify attention assumed
dense KV arrays but a quantized cache returns (packed, scales, biases)
tuples; and rollback crashed on `BatchQuantizedKVCache`.

**End state:**

- `models/cache.py`: `BatchQuantizedKVCache.is_trimmable()` → True with
  the same index-only `trim` as `BatchKVCache` (was False/no-op: rollback
  couldn't drop rejected draft tokens, and the exclusion-based
  `_is_ssm_cache` classifier misrouted it into gated-delta item
  assignment).
- `models/qwen3_5/language.py` target-verify branch — **final form** (the
  intermediate per-position tuple-slicing loop from `2714def` was
  replaced in `8ced71d` because it accumulated ~10 GB of temporaries and
  ate the speculative speedup): **one SDPA call with `mask="causal"`**
  when the mask is None/str (equivalence proven: fp16 bit-exact,
  quantized at float eps; measured decode 23.7 → 26.7 tok/s and −10 GB
  transient); the per-position loop (tuple-aware) remains only for
  explicit array masks.

### E2. Row-split prefill with quantized caches (`86099f4`)

**Why:** concurrent multi-row prefill with kv-quant crashed (and before
the crash, silently *discarded* row results). Only reachable when
`max_concurrent_requests` > 1 (see G1) — kept as defense-in-depth.

**End state:** `_extract_row_cache` hands rows of an *empty*
`BatchQuantizedKVCache` a matching `QuantizedKVCache` (was: plain fp16
`KVCache`); `BatchQuantizedKVCache.merge` raises a descriptive TypeError
on dense rows.

### E3. Peak-memory levers (`8ced71d`, config defaults in D1)

**Why:** bench peak was ~45 GB. Fully attributed via measurement:

1. **Prefill score matrices** — each chunk materializes
   step × context × heads scores per layer (~5.7 GB/layer at 4096×31k;
   identical for fused fp16 and quantized paths). → default
   `prefill_step_size` **1024**: cold-31.7k peak 36.9 → 28.6 GB at 980 →
   910 tok/s; warm workload unchanged.
2. **n-gram draft window doubling to 32** — each drafted block's verify
   materializes per-layer GDN intermediate states ∝ block length;
   transiently pinned ~11 GB during decode at 30k. → `ngram_max: 8`:
   same wall time on the replay, 40.8 → 30.1 GB peak.
3. (The third contributor was the verify loop, fixed in E1.)

Combined with F1, measured end state: bench peak 45.2 → **31.9 GB**,
decode 30.7 tok/s, accept 56%.

---

## Group F — Quantized APC session storage (core apc; only needed if `kv_quantization` is used)

### F1. Store and resume exact-session KV quantized (`444dc9a`)

**Why:** with kv-quant on, APC stored session anchors **dequantized to
fp16** and re-quantized them on every resume — double quantization error,
full fp16 materialization, and the `kv_quantization` toggle barely moved
real memory (sessions 4.05 GB either way).

**End state:** sessions hold (packed, scales, biases) tuples end to end:

- `apc_adapters.py`: `QuantizedKVCacheCloneAdapter` (clones without
  dequantizing) + empty-batch-quantized branch.
- `apc.py`: store-time quantization of fp16 clones following the live
  policy (`should_quantize_kv_layer` + `QUANTIZED_KV_START`, read from
  the same env the server uses — needed because the prompt batch can
  still be float at harvest time); session classification/slicing for
  tuples; tuple-aware `kv_bytes` stats; new `"quant_kv"` disk snapshot
  kind + U32/I32/I64 in the safetensors dtype map (U32 required; the
  int types also fix a latent gap for ArraysCache side arrays). Disk
  namespaces already key on kv-bits, so formats never mix.
- `models/cache.py`: `BatchQuantizedKVCache.merge` (tuple-layout mirror
  of `BatchKVCache.merge`) so warm rows join batches directly; the resume
  requantization becomes a passthrough.
- fp16 and TurboQuant snapshots unchanged. Session resume is bit-exact
  with the stored cache. Measured: sessions 4.05 → 2.27 GB, steady 28.6 →
  25.9 GB, seed restores quantized from disk in 0.4 s.
- Tests: 5 added; 2 existing tests updated that had locked in the fp16
  flattening.

**Pick notes:** independent of E except sharing
`BatchQuantizedKVCache.merge`'s file; needs E1's trim for MTP+kv to work
at all. Skip the whole group if kv_quantization won't be used.

---

## Group G — Robustness / incident response (added while chasing production incidents)

### G1. Serialize request processing (`0a35969`)

**Why (user decision):** the multi-request batching paths are
undertested with kv-quant (E2's crash was hit by real concurrent Junie
requests); rather than trust them, process strictly one request at a
time.

**End state:** `generation.py` `_collect_pending_requests` takes
`max_items`; the run loop computes capacity from
`MLX_VLM_MAX_CONCURRENT_REQUESTS` (0 = unlimited = stock behavior; core
default unchanged, batching tests intact). Junie config
`max_concurrent_requests: 1` — one request in the engine, the rest wait
in the queue (visible as `queue_depth`). MTP arrival-coalescing skipped
at cap 1. Verified: 3 concurrent requests → in_flight 3 / engine 1 /
queue 2, all succeed.

### G2. `/shutdown` cancels in-flight inference (`dde81a8`)

**Why:** uvicorn's graceful shutdown waits for open connections — stop
during a generation hung 39 s+.

**End state:** `/shutdown` flips phase to `stopping` (new busy phase:
new inference → 503), sweeps the generator's active uids through the
existing per-uid cancellation (clients get partial output), arms daemon
timers: SIGTERM at 0.5 s, SIGKILL fallback at 10 s. Measured: mid-decode
0.9 s, mid-prefill 1.1 s, idle 0.9 s.

### G3. Token-id-0 corruption bandage (`b6b9602`)

**Why:** corrupted model state (see "Open problem") generates endless
token id 0 ("!") — argmax of zeroed logits — and speculative decoding
bulk-accepts it, burning 32 k tokens per request.

**End state:** the generation loop tracks consecutive token-id-0
emissions per request (`MLX_VLM_MAX_ZERO_TOKEN_RUN`, default 8, 0
disables). On detection: ERROR log with request id, request fails with
`CorruptedGenerationError` → `/chat/completions` maps it to **503**
("model serving is restarting; retry") → `runtime.on_generation_corrupted`
hook → junie layer restarts model serving in the background. A runaway
now costs ~8 junk tokens + one reload instead of 32 k tokens. Unit-tested
via the fake-batch harness; battle-tested live (caught 7/8 corrupted
cycles, self-healed every time).

### G4. Guarded reload after auto-unload (`cadcb2e`)

**Why:** after an idle auto-unload, the next request loaded the model
**lazily inside its own handler**: no lock (double-load race), no busy
phase, no seed re-warmup (everything paid cold prefill), and — measured —
a corrupted model in ~90% of cycles.

**End state:** `get_cached_model` gains a text-model-cache-miss hook
(`runtime.on_text_model_load`, no-op when unregistered). The junie layer
redirects reloads of the *configured* model onto the same guarded worker
`apply_settings` uses (lock, `loading_model` phase in `/status`, seed
warmup, `loaded_at`); the triggering request answers 503 and the client
retries. Other models / other deployments keep stock lazy behavior.

### G5. Net auto-unload semantics (end state of A2+G3+G4)

Idle → watchdog unloads (memory to ~0.6 GB) → next request → 503 +
guarded background reload (~20 s warm) → seed re-warmed → retry succeeds.
If the reload comes up corrupted (see below), the bandage catches it at
~7 tokens and reloads again; convergence in ≈1.1 tries.

---

## ⚠️ Open problem (root cause NOT found): in-process model reload corruption

Loading a model into a process that previously ran and unloaded one
produces corrupted serving state (all-token-0 output) with probability
depending on the load path. Measured (2026-08-06, kv_quantization on):

| Load path | Corruption rate |
|---|---|
| Fresh process (kill + start) | **0/36** — always clean, all prefill sizes |
| In-process reload, guarded worker thread | ~1/13 |
| In-process reload, lazy on event-loop thread | 7/8 |

- Trigger frequency is driven by `auto_unload_time` (deliberately kept at
  10 s on this rig to catch incidents) and by `apply_settings` restarts.
- Thread-sensitivity + probabilistic nature suggests a cross-thread /
  cross-stream buffer reuse race (MLX streams are per-thread; the
  watchdog frees on its thread, the loader allocates on another).
  Unproven. Not established whether fp16 (kv off) reloads also corrupt.
- Next diagnostic (not done): standalone one-process
  load→unload→load→logit-check harness; bisect int8-prefill / kv-quant /
  drafter / APC.
- Current mitigation = G3 + G4 (self-healing). A rebuild that drops
  auto-unload/apply-settings restarts entirely (process restart instead)
  would sidestep the problem — fresh loads are provably clean.

## Pre-existing failures (NOT from this work)

At base `3e1ca5e`: 6 failures in `test_server.py` (test helpers build
`ResponseGenerator` via `__new__` and miss `_raw_token_log`; one of these
was fixed en passant) and 8 in `test_apc.py` (disk-scope=pinned default
vs old expectations). All other suites green throughout; this work added
tests (progress snapshot, serialization, corruption detection, quantized
APC ×5) — 248 passing in test_server.py at HEAD.

## Cherry-pick dependency sketch

```
A1 (control plane) ──┬── A2 (watchdog) ── G4/G5 (needs D1 for config field)
                     ├── D1 (config)  ── D2 (launcher) ── C1 (final start.sh)
                     ├── G2 (shutdown)
                     └── B2 (/status memory; core cache-reset fix is standalone)
B1, B3 ── standalone core fixes (keep in any variant)
E1 ── needed for kv_quantization at all;  E2, F1 build on kv usage
E3 ── config defaults only (needs D1) + the causal-verify part of E1
G1 ── core engine knob standalone; junie default needs D1
G3 ── core detection standalone (503 without auto-restart);
      auto-restart hook needs A1's worker
```
