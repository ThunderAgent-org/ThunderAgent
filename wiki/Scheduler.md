# Scheduler

The scheduler is the core of ThunderAgent's program-aware capacity management. It runs in **TR mode** (`--router tr`) and is responsible for balancing KV-cache usage across backends by pausing and resuming agentic programs.

## Program Lifecycle

Each program (identified by `program_id`) transitions through these states:

```
                  ┌────────────┐
    new request   │            │   request completed
   ─────────────► │  REASONING │ ─────────────────────►  ACTING
                  │  (on GPU)  │                        (off GPU,
                  │            │                     executing tool)
                  └─────┬──────┘                          │
                        │                                 │
                        │  capacity overflow               │  next request
                        │  (mark for pause)                │  arrives
                        ▼                                 ▼
                  ┌────────────┐                    ┌────────────┐
                  │   MARKED   │──── becomes ──────►│   PAUSED   │
                  │ FOR PAUSE  │     ACTING         │  (in queue) │
                  └────────────┘                    └─────┬──────┘
                                                         │
                                                         │  scheduler resumes
                                                         ▼
                                                   ┌────────────┐
                                                   │  REASONING  │
                                                   │  (resumed)  │
                                                   └────────────┘
```

### Status vs State

- **ProgramStatus** describes what the program is currently doing:
  - `REASONING` -- On GPU, running inference in the backend
  - `ACTING` -- Off GPU, executing a tool or waiting for the next request

- **ProgramState** describes the lifecycle:
  - `ACTIVE` -- Running normally on an assigned backend
  - `PAUSED` -- Waiting in the global paused queue for the scheduler to resume it
  - `TERMINATED` -- Released and cleaned up

## Capacity Model

The scheduler tracks KV-cache capacity per backend:

```
capacity_used = reasoning_tokens + (acting_token_weight * acting_tokens) - shared_tokens + buffer
```

Where:
- **reasoning_tokens**: Sum of `total_tokens` across all `REASONING` programs on the backend
- **acting_tokens**: Sum of `total_tokens` across all `ACTING` programs on the backend
- **acting_token_weight**: Configurable weight (default 1.0) for acting programs. Lower values assume acting programs' KV cache will be evicted before they return.
- **shared_tokens**: Prefix cache savings (difference between program-tracked tokens and actual vLLM KV-cache usage)
- **buffer**: `BUFFER_PER_PROGRAM (100)` tokens reserved per active program for decode headroom

A backend is **over capacity** when `capacity_used > total_tokens_capacity`.

## Periodic Scheduler

The scheduler runs a background loop every `--scheduler-interval` seconds (default 5s):

```
_scheduled_check():
  1. Fetch fresh metrics from all backends
  2. Resume waiting programs (_greedy_resume)
  3. Check each backend for capacity overflow (_pause_until_safe)
```

### Pause Logic (`_pause_until_safe`)

When a backend exceeds capacity, programs are paused in priority order:

1. **ACTING programs** (smallest tokens first) -- These are off-GPU and can be paused immediately
2. **REASONING programs** (smallest tokens first) -- These are on-GPU; they are *marked* for pause and will be paused when they transition to ACTING

Marking a REASONING program accounts for its tokens in `future_paused_tokens` so the scheduler doesn't over-pause.

### Resume Logic (`_greedy_resume`)

Paused programs are resumed using **Best Fit Decreasing (BFD)** bin-packing:

1. Compute remaining capacity across all healthy backends
2. Select programs to resume in priority order:
   - **REASONING programs** (step > 1) -- highest priority (have a pending request waiting)
   - **New programs** (step = 1) -- medium priority
   - **ACTING programs** -- lowest priority
3. Sort selected programs by token count (largest first)
4. Place each program on the backend with the most remaining capacity
5. Re-sort backends after each placement

This maximizes the number of programs that can run concurrently while respecting capacity constraints.

### Acting Token Decay

When `--use-acting-token-decay` is enabled, the resume logic uses an optimistic capacity estimate:

```
decayed_tokens = acting_tokens * 2^(-t)
```

Where `t` is seconds since the program entered ACTING state. Programs that have been acting longer are assumed more likely to have their KV cache evicted, freeing capacity for resumed programs.

## Token Estimation

ThunderAgent estimates token counts from payload character length using a global `char_to_token_ratio`:

- Initial value: 5.0 (1 token per 5 characters)
- Updated after each request with momentum: `ratio = 0.2 * sample + 0.8 * old_ratio`
- Used for capacity checks before the backend returns actual token counts

## Pause/Resume Coordination

When a program is paused:
1. It is unregistered from its backend
2. Added to the `global_waiting_queue` with metadata (`PausedInfo`)
3. An `asyncio.Event` is created; the request coroutine waits on it
4. When the scheduler resumes the program, it registers it with a target backend and sets the event

Timeout: If a program waits for more than 30 minutes, it is force-resumed to the least-loaded backend.

## Default Mode

When `--router default` is used, no scheduling occurs:
- New programs are assigned to the least-loaded backend (by program count)
- No capacity checks, no pause/resume
- Suitable for simple setups where capacity management is not needed
