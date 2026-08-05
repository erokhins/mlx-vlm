# Junie local server — HTTP API

All endpoints live on one HTTP server (default `http://localhost:8085`,
set by `PORT` in `start.sh`). The port never changes while the process is
alive: settings changes restart *model serving* behind the API, not the
HTTP server itself.

Every endpoint below is also available with a `/v1` prefix
(`/status` ≡ `/v1/status`, `/health` has no alias). When the server was
started with `--api-key`, all of them require
`Authorization: Bearer <key>`; without it (the `start.sh` default) no auth
is needed.

`./serverctl.sh` in the repo root is a thin curl wrapper over these
endpoints (`./serverctl.sh status`, `settings`,
`apply max_context_length=150000`, `wait`, `stop`, ...).

Inference itself goes through **`POST /v1/chat/completions`** — standard
OpenAI chat-completions API (not documented here). Its only local quirks:
responses carry an extra `timings` block, and while a model is being
(re)loaded it returns `503` (see [Errors](#errors)).

## Lifecycle

The server process starts serving HTTP within ~2 seconds; the model loads
in the background. `GET /status` reports the phase:

```
starting -> loading_model -> warming_up -> ready
                  ^                          |
                  +------ restarting <-------+   (POST /apply_settings)
```

`error` is a terminal phase for a failed load (`phase_detail` says why);
the HTTP server stays up so the state is observable and a new
`/apply_settings` can retry.

---

## Control plane

### `GET /status`

Lifecycle phase plus live inference progress.

While the model is loading:

```json
{
  "phase": "loading_model",
  "phase_detail": "loading mlx-community/Qwen3.6-27B-4bit",
  "phase_since_unix": 1785925146.161,
  "uptime_s": 2.5,
  "model": {
    "loaded": false,
    "id": "mlx-community/Qwen3.6-27B-4bit",
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "context_limit": null
  },
  "inference": {
    "in_progress": false,
    "in_flight": 0,
    "queue_depth": 0,
    "requests": []
  }
}
```

Ready, with one request mid-prefill:

```json
{
  "phase": "ready",
  "phase_detail": null,
  "phase_since_unix": 1785925149.289,
  "uptime_s": 512.3,
  "model": {
    "loaded": true,
    "id": "mlx-community/Qwen3.6-27B-4bit",
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "context_limit": 262144
  },
  "memory": {
    "total_gb": 19.06,
    "peak_gb": 21.49,
    "kv_cache_gb": 1.9
  },
  "inference": {
    "in_progress": true,
    "in_flight": 1,
    "queue_depth": 0,
    "requests": [
      {
        "request_id": "16b8f2e40",
        "stage": "prefill",
        "prompt_tokens": 24016,
        "prefill_processed": 16384,
        "generated_tokens": 0,
        "prefill_progress": 0.6822,
        "elapsed_s": 3.412
      }
    ]
  }
}
```

Fields:

- `phase` — `starting` / `loading_model` / `warming_up` / `ready` /
  `restarting` / `error`. `warming_up` means the model is loaded and
  already serving; the pinned seed prefix is being prefilled.
- `model.id` — the loaded model, or the configured one while loading.
- `model.context_limit` — effective limit: `min` of the model's native
  context and the configured `max_context_length`.
- `memory.total_gb` — the server process's physical footprint (same number
  Activity Monitor shows), including the model weights and all caches.
- `memory.peak_gb` — lifetime maximum of that footprint (worst case this
  run has needed).
- `memory.kv_cache_gb` — in-RAM KV held by the prefix cache (warm
  conversations + the pinned seed). This is the one part that can be
  freed without unloading the model: `POST /v1/cache/reset` releases it
  at the cost of the next requests re-prefilling their context.
- `inference.requests[]` — one entry per in-flight batched request:
  - `stage` — `queued` → `prefill` → `decode`.
  - `prefill_progress` — `prefill_processed / prompt_tokens`, 0–1
    (APC-cached tokens count as processed, so warm requests jump to ~1
    immediately).
  - `generated_tokens` — decoded tokens so far (grows during `decode`).
  - `elapsed_s` — since the request was queued.
- `inference.in_flight` — HTTP-level count (includes requests not yet in
  the batch, e.g. during tokenization); `in_progress` is the single
  "is it busy" boolean.

### `GET /current_settings`

The settings model serving is currently running with.

```json
{
  "model_name": "mlx-community/Qwen3.6-27B-4bit",
  "max_context_length": null,
  "kv_quantization": false,
  "auto_unload_time": null
}
```

- `max_context_length` — server-side prompt+generation token cap
  (`null` = unlimited, the model's native context applies).
- `kv_quantization` — whether the KV cache is quantized (8-bit).
- `auto_unload_time` — seconds of inference inactivity after which the
  model is unloaded from memory (`null` = never). After an auto-unload
  the server stays `ready` and `/status` shows `model.loaded: false`;
  the next inference request reloads the model (that request is slow).

### `POST /apply_settings`

Apply new serving settings. The HTTP server keeps running. Changing
`model_name`, `max_context_length` or `kv_quantization` restarts model
serving in the background (unload → apply → reload → seed re-warmup) —
**poll `GET /status` until `phase` is `"ready"`**; the reload of the 27B
model takes on the order of a minute. `auto_unload_time` alone applies
live, without a restart.

Request — any subset of:

| Field | Type | Effect |
|---|---|---|
| `model_name` | string | switch to a different model (HF repo id or local path) |
| `max_context_length` | int > 0 or `null` | token cap for prompt + generation; `null` removes the cap |
| `kv_quantization` | bool | turn 8-bit KV cache quantization on or off |
| `auto_unload_time` | int > 0 or `null` | idle seconds before the model is unloaded from memory; `null` disables |
| `force` | bool | proceed even when inference is in flight (aborts it) |

```json
{
  "max_context_length": 150000,
  "kv_quantization": true
}
```

Response (`200`) when a restart was started:

```json
{
  "status": "applying",
  "model": "mlx-community/Qwen3.6-27B-4bit",
  "changes": ["kv_quantization", "max_context_length"],
  "message": "Model serving is restarting; poll GET /status until phase is 'ready'."
}
```

Response (`200`) for a live change (`auto_unload_time` only):

```json
{
  "status": "applied",
  "changes": ["auto_unload_time"],
  "settings": {
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    "max_context_length": 150000,
    "kv_quantization": true,
    "auto_unload_time": 600
  }
}
```

Errors:

- `400` — validation, e.g.
  `{"detail": "\"max_context_length\" must be a positive integer or null."}` or
  `{"detail": "Unknown settings: ['bogus']"}`
- `409` — a settings change is already in progress, the server is still
  loading, or inference is in flight and `force` was not set:
  `{"detail": "1 inference request(s) in flight; pass \"force\": true to restart model serving anyway."}`

### `POST /shutdown`

Graceful shutdown of the whole server process (empty body).

```json
{"status": "shutting_down"}
```

The process exits within a few seconds; afterwards the port stops
answering. Start again with `./start.sh`.

---

## Monitoring

### `GET /health`

Cheap liveness probe (returns `200` as soon as HTTP is up — use `/status`
to know whether the model is actually ready).

```json
{
  "status": "healthy",
  "loaded_model": "mlx-community/Qwen3.6-27B-4bit",
  "loaded_adapter": null,
  "loaded_models": {
    "text_generation": {
      "model": "mlx-community/Qwen3.6-27B-4bit",
      "adapter": null,
      "model_kind": "text_generation"
    }
  },
  "loaded_context_size": 262144,
  "configured_context_limit": null,
  "effective_context_limit": 262144,
  "loaded_tool_parser": "qwen3_coder",
  "continuous_batching_enabled": true,
  "apc_enabled": true
}
```

### `GET /metrics`

Per-request serving metrics: `latest` (last completed request), `recent`
(rolling window of the same envelopes), `summary` (lifetime counters) and
`server` (runtime snapshot incl. APC stats).

```json
{
  "latest": {
    "timestamp_unix": 1785844283.393,
    "endpoint": "/chat/completions",
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "stream": false,
    "backend": "continuous_batching",
    "prompt_tokens": 24016,
    "completion_tokens": 517,
    "generated_tokens": 517,
    "total_tokens": 24533,
    "prompt_eval_time_s": 4.292,
    "prefill_tok_s": 5595.6,
    "ttft_s": 4.416,
    "decode_elapsed_s": 41.259,
    "request_elapsed_s": 45.713,
    "decode_tok_s": 12.53,
    "peak_memory_gb": 40.09,
    "finish_reason": "tool_calls",
    "tool_parser": "qwen3_coder",
    "tool_calls": true,
    "apc_enabled": true
  },
  "recent": ["... same envelope per recent request ..."],
  "summary": {
    "uptime_s": 82154.8,
    "requests_started": 33,
    "requests_completed": 33,
    "requests_failed": 0,
    "streaming_requests": 0,
    "in_flight": 0,
    "prompt_tokens_total": 730592,
    "completion_tokens_total": 5347,
    "generated_tokens_total": 5347,
    "avg_request_time_s": 14.62,
    "avg_request_tok_s": 11.08,
    "avg_decode_tok_s": 16.59,
    "last_request_at": 1785844283.393,
    "last_error": null
  },
  "server": {"... same shape as /health plus queue depths and full APC stats ..."}
}
```

### `GET /v1/cache/stats`

Automatic Prefix Cache statistics (`{"enabled": false}` when APC is off).

```json
{
  "enabled": true,
  "lookups_hit": 32,
  "lookups_miss": 0,
  "matched_tokens": 672172,
  "token_hit_rate": 1.0,
  "exact_hits": 32,
  "exact_stores": 63,
  "disk_hits": 3,
  "disk_writes": 0,
  "disk_bytes": 2281211630,
  "disk_files": 2,
  "exact_sessions": [
    {
      "anchor_tokens": 15557,
      "checkpoints": [15557],
      "pinned": true,
      "kv_bytes": 1019543552
    },
    {
      "anchor_tokens": 31713,
      "checkpoints": [26941, 26957, 30269, 30285, 31697, 31713],
      "pinned": false,
      "kv_bytes": 2078343168
    }
  ]
}
```

(`exact_sessions` = warm conversations; the `pinned: true` entry is the
cross-session seed prefix. A handful of low-level counters — `block_size`,
`pool_used`, `rejects`, ... — are omitted here for brevity.)

### `POST /v1/cache/reset`

Clear the APC (empty body). Response: `{"enabled": true, "status": "cleared"}`
(or `{"enabled": false}` when APC is off).

### `GET /v1/models`

OpenAI-style list of locally available models.

```json
{
  "object": "list",
  "data": [
    {"id": "mlx-community/Qwen3.6-27B-MTP-4bit", "object": "model", "created": 1782999397},
    {"id": "mlx-community/Qwen3.6-27B-4bit", "object": "model", "created": 1784723103}
  ]
}
```

### `POST /unload`

Unload all models from memory without restarting anything (empty body).
Mostly superseded by `/apply_settings`; a later request (or
`/apply_settings`) loads a model again.

```json
{
  "status": "success",
  "message": "Model unloaded successfully",
  "unloaded": {
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    "adapter_name": null,
    "models": {"...": "..."}
  }
}
```

---

## Errors

- **`503`** on inference endpoints while a model is being (re)loaded:

  ```json
  {"detail": "Model serving is unavailable: server phase is 'loading_model'. Poll GET /status and retry once the phase is 'ready'."}
  ```

- **`401`** `{"detail": "Invalid API key"}` — only when the server was
  started with `--api-key` and the `Authorization` header is missing or
  wrong.

- **Connection refused** — the process is not running (or was just
  started and HTTP isn't up yet, a ~2 s window). Start with `./start.sh`
  and check `mlx_server.log` if it doesn't come up.
