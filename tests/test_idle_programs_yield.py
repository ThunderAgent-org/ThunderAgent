"""Programs that stay ACTING must not keep a waiting request out forever.

The case: five programs finished without POST /programs/release and stay
ACTING, holding 197,064 of 208,768 tokens (numbers from a real SGLang run
with eight agents at once). Three programs whose next request has arrived
wait in the paused pool. Each needs about 45k tokens, so none fits and the
backend is not over capacity, so nothing is paused either.
"""

import asyncio
import time

from ThunderAgent.backend.sglang_metrics import SGLangCacheConfig
from ThunderAgent.program import Program, ProgramState, ProgramStatus
from ThunderAgent.scheduler.router import MultiBackendRouter

URL = "http://backend:30000/v1"
CAPACITY = 208_768
FINISHED = [40_000, 39_000, 38_000, 40_064, 40_000]  # 197,064 tokens
WAITING = [45_000, 44_000, 46_000]


def make_router(pause_idle_after: float) -> MultiBackendRouter:
    router = MultiBackendRouter(URL, backend_type="sglang", pause_idle_after=pause_idle_after)
    client = router.backends[URL].metrics_client
    client.cache_config = SGLangCacheConfig(total_tokens_capacity=CAPACITY)
    client.healthy = True

    async def fetch_metrics() -> bool:
        return True

    client.fetch_metrics = fetch_metrics
    return router


def add_acting(router: MultiBackendRouter, pid: str, tokens: int, idle_s: float) -> Program:
    """A running program between requests, idle for idle_s seconds."""
    program = Program(
        program_id=pid,
        backend_url=URL,
        status=ProgramStatus.ACTING,
        state=ProgramState.ACTIVE,
        total_tokens=tokens,
        step_count=20,
        acting_since=time.time() - idle_s,
    )
    router.programs[pid] = program
    router.backends[URL].register_program(pid, program)
    return program


def add_waiting(router: MultiBackendRouter, pid: str, tokens: int) -> Program:
    """A paused program whose next request has arrived and waits."""
    program = add_acting(router, pid, tokens, idle_s=1.0)
    router._pause_program(pid, program)
    # What update_program_before_request does when the request arrives.
    program.status = ProgramStatus.REASONING
    program.acting_since = None
    return program


def setup(pause_idle_after: float, finished_idle_s: float):
    router = make_router(pause_idle_after)
    finished = [
        add_acting(router, f"finished-{i}", tokens, idle_s=finished_idle_s + 60 * (len(FINISHED) - i))
        for i, tokens in enumerate(FINISHED)
    ]
    waiting = [add_waiting(router, f"waiting-{i}", tokens) for i, tokens in enumerate(WAITING)]
    return router, finished, waiting


def resumed(program: Program) -> bool:
    return program.state == ProgramState.ACTIVE and program.waiting_event is None


def run(coro):
    return asyncio.run(coro)


def test_idle_programs_hold_capacity_and_nothing_is_paused_without_the_fix():
    async def main():
        router, finished, waiting = setup(pause_idle_after=0, finished_idle_s=300)
        assert 0 <= router.backends[URL].remaining_capacity() < min(WAITING)
        for _ in range(5):
            await router._scheduled_check()
        assert not any(resumed(p) for p in waiting)
        assert all(p.state == ProgramState.ACTIVE for p in finished)
        await router.client.aclose()

    run(main())


def test_long_idle_programs_yield_to_waiting_requests():
    async def main():
        router, finished, waiting = setup(pause_idle_after=60, finished_idle_s=300)
        backend = router.backends[URL]

        # One tick: the longest-idle finished program is paused and the
        # smallest waiting program is placed.
        await router._scheduled_check()
        assert finished[0].state == ProgramState.PAUSED
        assert all(p.state == ProgramState.ACTIVE for p in finished[1:])
        assert resumed(waiting[1])  # 44,000 tokens, the smallest
        assert backend.remaining_capacity() >= 0

        # Two more ticks place the other two, never going over capacity.
        for _ in range(2):
            await router._scheduled_check()
            assert backend.remaining_capacity() >= 0
        assert all(resumed(p) for p in waiting)
        await router.client.aclose()

    run(main())


def test_recently_idle_programs_are_not_paused():
    async def main():
        router, finished, waiting = setup(pause_idle_after=600, finished_idle_s=0)
        # Idle for 60 to 300 s, all under 600 s.
        await router._scheduled_check()
        assert all(p.state == ProgramState.ACTIVE for p in finished)
        assert not any(resumed(p) for p in waiting)
        await router.client.aclose()

    run(main())


def test_idle_programs_stay_when_nothing_waits():
    async def main():
        router = make_router(pause_idle_after=60)
        finished = [add_acting(router, f"finished-{i}", t, idle_s=3600) for i, t in enumerate(FINISHED)]
        await router._scheduled_check()
        assert all(p.state == ProgramState.ACTIVE for p in finished)
        await router.client.aclose()

    run(main())


def test_no_pause_when_freeing_every_idle_program_is_not_enough():
    async def main():
        router = make_router(pause_idle_after=60)
        add_acting(router, "busy", 150_000, idle_s=1.0)  # idle too briefly to yield
        old = add_acting(router, "old", 30_000, idle_s=3600)
        waiting = add_waiting(router, "waiting", 60_000)
        await router._scheduled_check()
        # Pausing "old" frees 30k; with ~28k free that is still short of 60k.
        assert old.state == ProgramState.ACTIVE
        assert not resumed(waiting)
        await router.client.aclose()

    run(main())
