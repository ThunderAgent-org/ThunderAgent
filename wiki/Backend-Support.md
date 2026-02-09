# Backend Support

ThunderAgent supports three inference backends. Each backend has a dedicated metrics client that handles communication, metrics parsing, and capacity discovery.

## Supported Backends

| Backend | `--backend-type` | Metrics Format | Capacity Source |
|---------|-----------------|----------------|-----------------|
| [vLLM](https://github.com/vllm-project/vllm) | `vllm` | Prometheus text (`/metrics`) | `block_size * num_gpu_blocks` from `cache_config_info` |
| [SGLang](https://github.com/sgl-project/sglang) | `sglang` | Prometheus text (`/metrics`) + JSON (`/get_server_info`) | `max_total_num_tokens` from server info |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | `skyrl` | JSON (`/metrics`) | Estimated from engine count (200K tokens/engine default) |

## Architecture

All metrics clients implement the `MetricsClient` abstract interface:

```python
class MetricsClient(ABC):
    async def start_monitoring(self, interval: float) -> None
    async def stop_monitoring(self) -> None
    async def fetch_metrics(self) -> bool
    async def fetch_cache_config(self) -> bool
    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int
    def to_dict(self) -> dict
```

`BackendState` wraps a `MetricsClient` and adds program tracking (registered programs, capacity checks, token aggregation).

## vLLM Backend

### Metrics Parsing

Parses Prometheus text format from vLLM's `/metrics` endpoint:

| Metric | Description |
|--------|-------------|
| `vllm:num_requests_running` | Currently running requests |
| `vllm:num_requests_waiting` | Requests waiting in queue |
| `vllm:kv_cache_usage_perc` | KV cache utilization (0.0-1.0) |
| `vllm:prefix_cache_queries_total` | Cumulative prefix cache queries |
| `vllm:prefix_cache_hits_total` | Cumulative prefix cache hits |
| `vllm:prompt_tokens_total` | Cumulative prompt tokens processed |
| `vllm:generation_tokens_total` | Cumulative generation tokens |
| `vllm:num_preemptions_total` | Cumulative preemptions |
| `vllm:request_success_total` | Completed requests by finish reason |

### Cache Config

Extracted from `vllm:cache_config_info` labels:
- `block_size`: Tokens per KV cache block
- `num_gpu_blocks`: Total GPU blocks allocated
- `total_tokens_capacity = block_size * num_gpu_blocks`

### Shared Tokens Calculation

```
shared_tokens = reasoning_program_tokens - (kv_cache_usage_perc * total_tokens_capacity)
```

This captures prefix cache savings -- tokens shared across programs that don't consume additional KV cache.

## SGLang Backend

### Metrics Parsing

Combines two endpoints:
- `/metrics` (Prometheus text): Runtime metrics
- `/get_server_info` (JSON): Static capacity information

| Metric | Description |
|--------|-------------|
| `sglang:num_running_reqs` | Currently running requests |
| `sglang:num_queue_reqs` | Requests waiting in queue |
| `sglang:token_usage` | Token usage fraction |
| `sglang:cache_hit_rate` | Cache hit rate |
| `sglang:num_used_tokens` | Number of used tokens |
| `sglang:prompt_tokens_total` | Cumulative prompt tokens |
| `sglang:generation_tokens_total` | Cumulative generation tokens |

### Capacity Discovery

Extracts `max_total_num_tokens` from `/get_server_info` response. Falls back to checking `internal_states[].memory_usage.token_capacity`.

## SkyRL Backend

### Metrics Parsing

SkyRL exposes a JSON endpoint at `/metrics` with per-engine data:

```json
{
  "timestamp": 1770532792.69,
  "engines": [
    {
      "engine_id": 0,
      "kv_cache_usage_pct": 5.9,
      "num_running_reqs": 2,
      "num_waiting_reqs": 0,
      "num_cumulative_preemption": 0
    }
  ]
}
```

The client aggregates across all engines:
- Running/waiting requests: summed
- KV cache usage: averaged across engines
- Preemptions: summed

### Capacity Estimation

SkyRL doesn't expose exact token capacity. The client estimates 200,000 tokens per engine as a default, which works because the scheduler primarily uses `kv_cache_usage_perc` for decisions.

## Metrics Monitoring

When `--metrics` is enabled, each backend starts a background monitoring loop:

1. Fetch cache config once at startup
2. Poll `/metrics` every `--metrics-interval` seconds (default 5s)
3. Store the last 12 metrics samples in a history ring buffer
4. Update `healthy` status based on successful fetches

Metrics are used by:
- The scheduler for capacity calculations (`shared_tokens`, `kv_cache_usage`)
- The `/metrics` API endpoint for external monitoring
- The `/health` endpoint for system status
