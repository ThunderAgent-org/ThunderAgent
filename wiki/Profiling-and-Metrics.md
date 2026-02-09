# Profiling and Metrics

ThunderAgent provides two observability systems: **profiling** (per-program timing metrics) and **metrics monitoring** (per-backend KV-cache and request stats).

## Profiling

Enable with `--profile`. Optionally set the output directory with `--profile-dir` (default: `/tmp/thunderagent_profiles`).

### What is Tracked

For each inference step (request/response cycle) of every program, ThunderAgent records:

| Metric | Description |
|--------|-------------|
| `prefill_time` | Time from request start (after any pause) to first token received |
| `decode_time` | Time from first token to last token |
| `pause_time` | Time spent waiting in the paused queue before the request starts |
| `tool_call_time` | Time between the end of the previous request and the arrival of this request (tool execution time) |
| `kv_hit_rate` | Ratio of cached tokens to total prompt tokens (`cached_tokens / prompt_tokens`) |

### Timing Diagram

```
Previous request ends                    Current request
        │                                     │
        ├── tool_call_time ──►│◄── pause_time ──►│◄── prefill_time ──►│◄── decode_time ──►│
        │                     │                  │                    │                   │
   last_request_end     request_arrive     request_start        first_token          last_token
```

### CSV Export

Per-step metrics are automatically written to `{profile_dir}/step_profiles.csv`:

```csv
program_id,step_id,prefill_s,decode_s,pause_s,tool_call_s,kv_hit_rate,completed_at
agent-1,1,0.1523,2.3456,0.0,0.0,,1770532800.0
agent-1,2,0.0821,1.8734,0.0,1.2345,0.85,1770532805.0
agent-2,1,0.2011,3.1234,0.5123,0.0,,1770532803.0
```

Notes:
- `kv_hit_rate` is empty for the first step or when cached token info is unavailable
- `tool_call_time` is 0 for the first step (no previous request)
- All times are in seconds, rounded to 4 decimal places

### API Endpoints

- `GET /profiles` -- All program profiles (averages)
- `GET /profiles/{program_id}` -- Single program profile
- `GET /programs` -- Includes profile data when profiling is enabled

Response format:

```json
{
  "step_count": 5,
  "avg_prefill_s": 0.15,
  "avg_decode_s": 2.3,
  "avg_pause_s": 0.0,
  "avg_tool_call_s": 1.2,
  "avg_kv_hit_rate": 0.85
}
```

## Metrics Monitoring

Enable with `--metrics`. Configure the polling interval with `--metrics-interval` (default: 5.0s).

### What is Monitored

ThunderAgent polls each backend's metrics endpoint and tracks:

**vLLM Metrics:**
- Requests running and waiting
- KV cache usage percentage
- Prefix cache hit rate (queries and hits)
- Cumulative prompt and generation tokens
- Preemption count
- Request completion counts by finish reason (stop, length, abort, error)

**SGLang Metrics:**
- Requests running and waiting
- Token usage and cache hit rate
- Used token count
- Cumulative prompt and generation tokens

**SkyRL Metrics:**
- Requests running and waiting (aggregated across engines)
- Average KV cache usage percentage
- Preemption count
- Engine count

### Metrics History

Each backend maintains a ring buffer of the last 12 metrics samples. This is used internally for trend analysis and exposed via the `/metrics` API.

### API Endpoint

`GET /metrics` returns the current state of all backends:

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
      "shared_tokens": 5000,
      "capacity_overflow": 0,
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
