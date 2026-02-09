# Architecture

ThunderAgent acts as a proxy layer between agentic clients and inference backends. It intercepts OpenAI-compatible API calls, tracks per-program state, and schedules requests across one or more GPU backends to maximize KV-cache utilization and throughput.

## High-Level Design

```
┌─────────────────────┐
│   Agent Clients     │  (OpenAI-compatible API + program_id)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    ThunderAgent     │
│  ┌───────────────┐  │
│  │  FastAPI App  │  │  Receives /v1/chat/completions
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │  Scheduler    │  │  Program-aware routing, pause/resume
│  │  (Router)     │  │  Capacity-based scheduling (TR mode)
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Backend State │  │  Per-backend token tracking, metrics
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │Program Tracker│  │  Per-program status, lifecycle, profiling
│  └───────────────┘  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Inference Engines   │  vLLM / SGLang (serving)
│ (one or more GPUs)  │  SkyRL (RL rollout via vLLM async engine)
└─────────────────────┘
```

## Module Structure

```
ThunderAgent/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point (argument parsing, uvicorn launch)
├── app.py               # FastAPI application and route definitions
├── config.py            # Global configuration dataclass
├── backend/
│   ├── __init__.py
│   ├── metrics_base.py      # Abstract MetricsClient interface
│   ├── vllm_metrics.py      # vLLM Prometheus metrics parser & client
│   ├── sglang_metrics.py    # SGLang metrics parser & client
│   ├── skyrl_metrics.py     # SkyRL JSON metrics parser & client
│   └── state.py             # BackendState: capacity tracking, program registry
├── program/
│   ├── __init__.py
│   └── state.py             # Program, ProgramStatus, ProgramState
├── profile/
│   ├── __init__.py
│   └── state.py             # ProfileState, StepMetrics, CSV export
└── scheduler/
    ├── __init__.py
    ├── router.py             # MultiBackendRouter: scheduling, pause/resume, BFD
    └── vllm_request_processor.py  # HTTP forwarding, SSE parsing, usage extraction
```

## Core Components

### 1. FastAPI Application (`app.py`)

The entry point for all HTTP traffic. Defines routes for:
- `POST /v1/chat/completions` -- Main inference proxy endpoint
- `GET /v1/models` -- Model listing (proxied to backend)
- `GET /programs` -- List all tracked programs
- `POST /programs/release` -- Release a program's resources
- `GET /health` -- Health check with program stats
- `GET /profiles` and `GET /profiles/{program_id}` -- Profiling data
- `GET /metrics` -- Backend metrics

### 2. MultiBackendRouter (`scheduler/router.py`)

The central orchestrator. Responsibilities:
- **Program lifecycle**: Create, track, pause, resume, and release programs
- **Backend assignment**: Assign new programs to backends based on capacity
- **Capacity scheduling** (TR mode): Periodic scheduler that pauses overloaded backends and resumes waiting programs using Best Fit Decreasing (BFD) bin-packing
- **Request proxying**: Forward requests to the assigned backend, handle streaming/non-streaming responses
- **Token estimation**: Maintain a global `char_to_token_ratio` for estimating token counts from payload size

### 3. BackendState (`backend/state.py`)

Tracks the state of a single inference backend:
- Registered programs and their token counts
- Capacity calculations (reasoning tokens, acting tokens, shared tokens, buffer)
- Delegates metrics fetching to a `MetricsClient` implementation

### 4. Program State (`program/state.py`)

Each agentic program (identified by `program_id`) has:
- **ProgramStatus**: `REASONING` (on GPU, running inference) or `ACTING` (off GPU, executing tool)
- **ProgramState**: `ACTIVE`, `PAUSED`, or `TERMINATED`
- Token counts, backend assignment, and pause/resume coordination via `asyncio.Event`

### 5. ProfileState (`profile/state.py`)

Per-program timing instrumentation tracking:
- Prefill time (request start to first token)
- Decode time (first token to last token)
- Pause time (time spent waiting in queue)
- Tool-call time (time between requests, i.e., tool execution)
- KV cache hit rate per step
- Automatic CSV export of per-step metrics

## Request Flow

1. Client sends `POST /v1/chat/completions` with `program_id` in the payload
2. `app.py` extracts `program_id` and calls `router.get_or_create_program()`
3. `router.update_program_before_request()` handles:
   - **Default mode**: Assigns to least-loaded backend, no capacity checks
   - **TR mode**: Checks capacity, may pause the program and wait for the scheduler to resume it
4. Request is forwarded to the assigned backend via `proxy_request()`
5. Response usage info is extracted; `router.update_program_after_request()` transitions the program to `ACTING` status
6. The periodic scheduler (`_scheduled_check`) runs every N seconds:
   - Fetches fresh metrics from all backends
   - Resumes waiting programs via BFD bin-packing (`_greedy_resume`)
   - Pauses programs on overloaded backends (`_pause_until_safe`)
