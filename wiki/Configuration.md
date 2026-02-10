# Configuration

ThunderAgent is configured via command-line arguments passed to the `thunderagent` CLI or `python -m ThunderAgent`.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind the HTTP server to |
| `--port` | `8300` | Port to bind the HTTP server to |
| `--log-level` | `info` | Log level (`debug`, `info`, `warning`, `error`) |
| `--backends` | `http://localhost:8000` | Comma-separated list of backend URLs |
| `--backend-type` | `vllm` | Backend type: `vllm`, `sglang` (inference serving), or `skyrl` (RL rollout integration) |
| `--router` | `tr` | Router mode: `default` (pure proxy) or `tr` (capacity scheduling) |
| `--profile` | disabled | Enable per-program profiling (timing metrics) |
| `--profile-dir` | `/tmp/thunderagent_profiles` | Directory for profile CSV output |
| `--metrics` | disabled | Enable backend metrics monitoring |
| `--metrics-interval` | `5.0` | Interval in seconds between metrics fetches |
| `--scheduler-interval` | `5.0` | Interval in seconds between scheduler checks |
| `--acting-token-weight` | `1.0` | Weight for acting tokens in capacity calculation |
| `--use-acting-token-decay` | disabled | Use `2^(-t)` decay for acting tokens in resume capacity |

## Router Modes

### `default` -- Pure Proxy

- No capacity checks or scheduling
- Assigns new programs to the least-loaded backend (by program count)
- Suitable for simple single-backend setups or when capacity management is not needed

### `tr` -- Capacity Scheduling (ThunderRouter)

- Program-aware capacity scheduling across backends
- Tracks KV-cache usage per backend via metrics monitoring
- Pauses programs when backends are overloaded; resumes them using Best Fit Decreasing bin-packing
- Requires `--metrics` to be enabled for accurate capacity tracking

## Acting Token Weight

The `--acting-token-weight` controls how acting (tool-executing) programs are weighted in capacity calculations:

- `1.0` (default): Acting programs count at full weight. Conservative -- assumes tool execution returns quickly.
- `0.1`: Acting programs count at 10% weight. Aggressive -- assumes most acting programs' KV cache will be evicted before they return.
- `0.0`: Acting programs are ignored in capacity calculations.

## Acting Token Decay

When `--use-acting-token-decay` is enabled, the resume logic uses exponential decay (`2^(-t)`) on acting token weights, where `t` is the number of seconds a program has been in ACTING state. This provides an optimistic estimate of available capacity for resuming paused programs.

## Config Dataclass

Internally, configuration is stored in a `Config` dataclass:

```python
@dataclass
class Config:
    backends: List[str]              # Backend URLs
    router_mode: str                 # "default" or "tr"
    backend_type: str                # "vllm", "sglang" (serving), or "skyrl" (RL rollout)
    profile_enabled: bool            # Enable profiling
    profile_dir: str                 # CSV output directory
    metrics_enabled: bool            # Enable metrics monitoring
    metrics_interval: float          # Metrics fetch interval (seconds)
    scheduler_interval: float        # Scheduler check interval (seconds)
    acting_token_weight: float       # Weight for acting tokens
    use_acting_token_decay: bool     # Exponential decay on acting tokens
```

The config is set once at startup (in `__main__.py`) and accessed globally via `get_config()`.
