# Junie local server — install & run (from sources)

Serves `mlx-community/Qwen3.6-27B-4bit` on Apple Silicon as an
OpenAI-compatible endpoint for Junie, with MTP + n-gram speculative
decoding, prefix caching (APC) with a pinned cross-session seed, and int8
NAX prefill.

## 1. Clone

```bash
git clone -b local/serving-patches https://github.com/erokhins/mlx-vlm
cd mlx-vlm
```

## 2. Start (installs everything on first run)

```bash
./start.sh
```

The script returns immediately: it launches everything in the background
and prints nothing. All output goes to `mlx_server.log` in the repo dir
(gitignored, truncated on each start) — so after a problem you can always
inspect or share the log of the last run. Watch startup progress with:

```bash
curl -s http://localhost:8085/status   # phase: loading_model -> warming_up -> ready
```

The first run takes a while; it automatically:

1. downloads the model weights (~17 GB, SHA256-verified, resumable — just
   re-run the script if interrupted),
2. writes the Junie model descriptor and sets it as Junie's default model,
3. creates the Python virtualenv and installs dependencies (via `uv`,
   which is itself auto-installed and downloads Python 3.13 if the machine
   has none — the stock macOS `python3` is too old for `mlx>=0.32`),
4. starts the server and prefills + pins the shared Junie prompt prefix
   (~12 s; later restarts restore it from disk in ~0.5 s — look for
   `Seed prefix warmed and pinned` in the log).

Every later run skips 1–3 automatically (the descriptor/default are just
rewritten), so `./start.sh` is also the everyday start command; it's a
no-op when the server is already up. Stop with:

```bash
curl -s -X POST http://localhost:8085/shutdown
```

(`./start.sh --foreground` runs it in the current shell instead —
output still goes to the log; stop with Ctrl-C.)

Then restart Junie — it will use the local model by default. (If
`~/.junie/settings.json` didn't exist yet, start Junie once and re-run
`./start.sh`, or pick `mlx-community/Qwen3.6-27B-4bit` manually.)

## 3. Benchmark (optional)

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
| `<repo>/mlx_server.log` | log of the current/last server run (gitignored) |
| `<repo>/research/junie.json` | seed request: the stable Junie prompt prefix (system message + tool schemas + first user message), prefilled and pinned at startup |
| `<repo>/research/junie-replay/` | captured session + replay script used by `bench.sh` |
| `~/.local/share/junie-local/models/` | model weights, HF-hub layout (`models--mlx-community--Qwen3.6-27B-4bit`, `...-MTP-4bit`, plus `.models--*.installed` completion markers) |
| `~/.local/share/junie-local/incomplete_downloads/` | in-progress downloads (kept for resume, removed when done) |
| `~/.local/share/junie-local/apc-cache/` | APC disk tier — holds only the pinned seed snapshot (~1 GB) so it survives restarts |
| `<repo>/.uv/bin/uv` | `uv` binary (only when not already installed on the machine) |
| `<repo>/.uv/python/` | uv-managed CPython 3.13 (only when the machine has no suitable Python) |
| `~/.junie/models/local-qwen3.6-27b-4bit-vlm.json` | Junie model descriptor pointing at this server |
| `~/.junie/settings.json` | existing Junie settings; `modelForLaunch` is set to this model |

## Server endpoints

Full request/response examples for every endpoint: [JUNIE_API.md](JUNIE_API.md).
`./serverctl.sh` wraps them for the command line (`status`, `settings`,
`apply key=value`, `wait`, `stop`, ...).

| URL | What |
|---|---|
| `http://localhost:8085/v1/chat/completions` | OpenAI-compatible chat endpoint (this is what Junie calls); responses include a `timings` block with prefill/decode speeds and speculative-acceptance counters |
| `http://localhost:8085/health` | liveness check |
| `http://localhost:8085/status` | lifecycle phase (`loading_model` / `warming_up` / `ready` / `restarting` / `error`) plus live inference progress: per-request stage, prefill %, generated tokens |
| `http://localhost:8085/current_settings` | the settings model serving currently runs with (port, model, context size, KV cache quantization, ...) |
| `http://localhost:8085/apply_settings` | POST a JSON subset of `{model, context_size, kv_cache_quantization, kv_bits, kv_quant_scheme, kv_group_size, quantized_kv_start, max_tokens, force}` — restarts model serving (not the HTTP server) with the new settings; poll `/status` until `ready` |
| `http://localhost:8085/shutdown` | POST — graceful shutdown of the whole server process |
| `http://localhost:8085/v1/cache/stats` | APC stats: sessions, checkpoints, the pinned seed, hit counters |

## What to expect in the log

- `Seed prefix warmed and pinned: ... cached_tokens=14551 elapsed=0.5s` —
  the cross-session prefix is ready; new Junie sessions warm-start.
- `Prefill completed: ... cached_tokens=...` — APC hit size per request.
- `Speculative decode: ... accepted_tokens_per_round=... ngram_rounds=...`
  — MTP drafter + n-gram prompt-lookup acceptance per request.
- `Raw generated tokens: ...` — each response's tokens as text, with
  accepted MTP drafts in green and accepted n-gram drafts in cyan
  (`--log-raw-tokens`; view with `tail -f` or `less -R`).

## Tuning knobs (already set to measured optima)

All optional; see `start.sh` comments and `research/mtp-overhead/README.md`
for the measurements behind the defaults:

- `MLX_VLM_NGRAM_*` — n-gram prompt-lookup drafting (base window 4,
  full-accept doubling to 32; `MLX_VLM_NGRAM_DRAFT=0` disables).
- `APC_EXACT_SESSIONS` / `APC_SESSION_CHECKPOINTS` — warm-conversation
  capacity.
- `APC_DISK_EXACT_SCOPE=all` — persist every conversation snapshot to disk
  (default `pinned` keeps only the seed).
- `MLX_VLM_INT8_SCOPE=mlp` or dropping `--int8-prefill` — fallback if
  prefill quality issues ever show up.
