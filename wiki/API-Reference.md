# API Reference

ThunderAgent exposes an OpenAI-compatible API plus additional management endpoints. All endpoints are served by a FastAPI application.

## Inference Endpoints

### `POST /v1/chat/completions`

Main inference endpoint. Proxies chat completion requests to the assigned backend.

**Request Body**: Standard OpenAI chat completions payload, with an additional `program_id` field.

```json
{
  "model": "Qwen/Qwen3-32B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "program_id": "my-agent-1",
  "extra_body": {
    "program_id": "my-agent-1"
  }
}
```

`program_id` can be provided either as a top-level field or inside `extra_body` (the latter is compatible with the OpenAI Python client's `extra_body` parameter). If omitted, defaults to `"default"`.

**Response**: Standard OpenAI chat completions response (streaming or non-streaming). The `program_id` is stripped before forwarding to the backend.

**Behavior**:
- Creates a new program if `program_id` is not yet tracked
- In TR mode, may pause the request until capacity is available
- Updates program token counts from the response usage info
- Transitions program status from `REASONING` to `ACTING` after response

### `GET /v1/models`

Lists available models. Proxied to the first backend.

**Response**: Standard OpenAI models list response.

## Program Management

### `GET /programs`

Lists all tracked programs with their current state.

**Response**:

```json
{
  "my-agent-1": {
    "backend": "http://localhost:8000",
    "context_len": 4096,
    "total_tokens": 1234,
    "step_count": 5,
    "status": "acting",
    "state": "active",
    "profile": {
      "step_count": 5,
      "avg_prefill_s": 0.15,
      "avg_decode_s": 2.3,
      "avg_pause_s": 0.0,
      "avg_tool_call_s": 1.2,
      "avg_kv_hit_rate": 0.85
    }
  }
}
```

Fields:
- `backend`: Assigned backend URL
- `context_len`: Estimated context length (characters)
- `total_tokens`: Total tokens tracked for this program
- `step_count`: Number of inference steps completed
- `status`: Current activity -- `reasoning` or `acting`
- `state`: Lifecycle state -- `active`, `paused`, or `terminated`
- `profile`: Timing metrics (only if `--profile` is enabled)

### `POST /programs/release`

Releases a program and frees its tracked resources.

**Request Body**:

```json
{
  "program_id": "my-agent-1"
}
```

**Response**:

```json
{
  "program_id": "my-agent-1",
  "released": true
}
```

## Monitoring

### `GET /health`

Health check endpoint with system overview.

**Response**:

```json
{
  "status": "ok",
  "router_mode": "tr",
  "scheduling_enabled": true,
  "backends": ["http://localhost:8000"],
  "programs_count": 10,
  "reasoning_count": 3,
  "acting_count": 5,
  "paused_count": 2,
  "per_backend": {
    "http://localhost:8000": {
      "total": 8,
      "reasoning": 3,
      "acting": 5,
      "paused": 2,
      "marked_for_pause": 0,
      "future_paused_tokens": 0
    }
  },
  "profile_enabled": true
}
```

### `GET /metrics`

Returns backend metrics from all backends.

**Response**:

```json
{
  "metrics_enabled": true,
  "metrics_interval": 5.0,
  "backends": {
    "http://localhost:8000": {
      "url": "http://localhost:8000",
      "healthy": true,
      "monitoring": true,
      "active_program_tokens": 50000,
      "reasoning_program_tokens": 30000,
      "acting_program_tokens": 20000,
      "active_program_count": 8,
      "active_program_tokens_ratio": 0.4,
      "cache_config": {
        "block_size": 16,
        "num_gpu_blocks": 27283,
        "total_tokens_capacity": 436528
      },
      "metrics": {
        "num_requests_running": 3,
        "num_requests_waiting": 0,
        "kv_cache_usage_perc": 0.35,
        "prefix_cache_hit_rate": 0.82
      }
    }
  }
}
```

## Profiling

### `GET /profiles`

Lists profiling data for all programs. Returns 400 if profiling is not enabled.

**Response**:

```json
{
  "my-agent-1": {
    "step_count": 5,
    "avg_prefill_s": 0.15,
    "avg_decode_s": 2.3,
    "avg_pause_s": 0.0,
    "avg_tool_call_s": 1.2,
    "avg_kv_hit_rate": 0.85
  }
}
```

### `GET /profiles/{program_id}`

Returns profiling data for a specific program. Returns 404 if the program or profile is not found.
