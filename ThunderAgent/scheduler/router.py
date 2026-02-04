"""Router with program state tracking - supports multiple backends."""
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple

import httpx
from fastapi.responses import Response, StreamingResponse

from ..backend import BackendState
from ..program import ProgramState, ProgramStatus
from ..profile.state import ProfileState
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PausedInfo:
    """Metadata for a paused program stored in the global paused pool."""
    program_id: str
    total_tokens: int
    paused_at: float
    origin_backend: str
    step_count: int


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    def __init__(
        self, 
        backend_urls: str | List[str], 
        *, 
        profile_enabled: bool = False,
        scheduling_enabled: bool = True,
    ) -> None:
        # Support single URL string or list of URLs
        if isinstance(backend_urls, str):
            backend_urls = [url.strip() for url in backend_urls.split(",") if url.strip()]
        
        # All backends
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        
        # All programs (single source of truth)
        # Key: program_id, Value: ProgramState (which includes backend_url)
        self.programs: Dict[str, ProgramState] = {}

        # Global paused pool shared across backends (program_id -> PausedInfo).
        self.paused_pool: Dict[str, PausedInfo] = {}

        # Lock for atomic claim (select + pop) from paused_pool.
        self.pause_resume_lock = asyncio.Lock()
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self._resume_monitor_task: Optional[asyncio.Task] = None
        self._resume_monitor_stop = False

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
        # Always fetch cache config (needed for active_program_tokens_ratio)
        for backend in self.backends.values():
            await backend.fetch_cache_config()
        
        # Start metrics monitoring on each backend if enabled
        config = get_config()
        if config.metrics_enabled:
            for backend in self.backends.values():
                await backend.start_monitoring(config.metrics_interval)

        # Resume monitor (TR mode): periodically attempt to resume paused programs.
        if self.scheduling_enabled:
            self._resume_monitor_stop = False
            self._resume_monitor_task = asyncio.create_task(self._resume_monitor_loop(interval_sec=1.0))

    async def stop(self):
        """Stop the router."""
        # Stop resume monitor
        if self._resume_monitor_task is not None:
            self._resume_monitor_stop = True
            self._resume_monitor_task.cancel()
            try:
                await self._resume_monitor_task
            except asyncio.CancelledError:
                pass
            self._resume_monitor_task = None

        # Stop metrics monitoring on each backend
        for backend in self.backends.values():
            await backend.stop_monitoring()
        
        await self.client.aclose()
        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def get_backend(self, url: str) -> Optional[BackendState]:
        """Get a backend by URL."""
        return self.backends.get(url)

    def get_default_backend(self) -> BackendState:
        """Get the first backend (for simple single-backend usage)."""
        return next(iter(self.backends.values()))

    def select_backend_for_new_program(self) -> BackendState:
        """Select the least loaded backend for a new program."""
        # Count programs per backend
        #### TODO change backend assign logistics
        backend_load: Dict[str, int] = {url: 0 for url in self.backends}
        for state in self.programs.values():
            if state.backend_url in backend_load:
                backend_load[state.backend_url] += 1
        
        # Find the backend with least programs (only consider healthy ones)
        min_load = float('inf')
        best_backend = None
        for url, load in backend_load.items():
            backend = self.backends[url]
            if backend.healthy and load < min_load:
                min_load = load
                best_backend = backend
        
        # Fallback to first backend if all unhealthy
        return best_backend or self.get_default_backend()

    # -------------------------------------------------------------------------
    # Program State Management
    # -------------------------------------------------------------------------

    @staticmethod
    def _acting_token_weight(now: float, acting_since: Optional[float]) -> float:
        """Weight for ACTING program tokens: 2^(-t), where t is tool execution time in seconds."""
        if acting_since is None:
            return 1.0
        elapsed = max(0.0, float(now) - float(acting_since))
        # 2^(-t) = exp(-t * ln 2)
        return math.exp(-elapsed * math.log(2.0))

    def _effective_backend_load(self, backend_url: str, *, now: Optional[float] = None) -> Tuple[float, int, float, float]:
        """Return (effective_active_tokens, active_count, reasoning_tokens, weighted_acting_tokens)."""
        ts = float(now) if now is not None else time.time()
        reasoning_tokens = 0.0
        weighted_acting_tokens = 0.0
        active_count = 0
        for state in self.programs.values():
            if state.backend_url != backend_url:
                continue
            if state.status == ProgramStatus.REASONING:
                reasoning_tokens += float(state.total_tokens)
                active_count += 1
            elif state.status == ProgramStatus.ACTING:
                weighted_acting_tokens += float(state.total_tokens) * self._acting_token_weight(ts, state.acting_since)
                active_count += 1
        return reasoning_tokens + weighted_acting_tokens, active_count, reasoning_tokens, weighted_acting_tokens

    def _backend_capacity_overflow(self, backend: BackendState, *, include_future_release: bool) -> float:
        effective_tokens, active_count, _r, _a = self._effective_backend_load(backend.url)
        return backend.capacity_overflow_for(
            active_tokens=effective_tokens,
            active_count=active_count,
            include_future_release=include_future_release,
        )

    def _backend_has_capacity(
        self,
        backend: BackendState,
        *,
        extra_tokens: float = 0.0,
        extra_count: int = 0,
    ) -> bool:
        effective_tokens, active_count, _r, _a = self._effective_backend_load(backend.url)
        return backend.has_capacity_for(
            active_tokens=effective_tokens,
            active_count=active_count,
            extra_tokens=extra_tokens,
            extra_count=extra_count,
        )

    def _count_resumable_paused(self) -> int:
        """Count paused programs eligible for resume (tool_finished=True)."""
        n = 0
        for pid in self.paused_pool.keys():
            state = self.programs.get(pid)
            if not state or state.status != ProgramStatus.PAUSED:
                continue
            if not state.tool_finished:
                continue
            n += 1
        return n

    def get_effective_backend_stats(self) -> Dict[str, Any]:
        """Return per-backend effective load/capacity stats used by TR scheduling."""
        result: Dict[str, Any] = {}
        for url, backend in self.backends.items():
            effective_tokens, active_count, reasoning_tokens, weighted_acting_tokens = self._effective_backend_load(url)
            required = backend.required_tokens(active_tokens=effective_tokens, active_count=active_count)
            cap = backend.cache_config.total_tokens_capacity if backend.cache_config else None
            overflow = backend.capacity_overflow_for(
                active_tokens=effective_tokens,
                active_count=active_count,
                include_future_release=False,
            )
            overflow_future = backend.capacity_overflow_for(
                active_tokens=effective_tokens,
                active_count=active_count,
                include_future_release=True,
            )
            result[url] = {
                "effective_active_tokens": effective_tokens,
                "effective_active_count": active_count,
                "effective_reasoning_tokens": reasoning_tokens,
                "effective_weighted_acting_tokens": weighted_acting_tokens,
                "required_tokens": required,
                "total_tokens_capacity": cap,
                "capacity_overflow": overflow,
                "capacity_overflow_include_future_release": overflow_future,
                "future_paused_tokens": backend.future_paused_tokens,
                "shared_token": BackendState.shared_token,
            }
        return result

    async def _resume_monitor_loop(self, *, interval_sec: float) -> None:
        """Every interval, try to resume paused programs on each backend (best-effort)."""
        while not self._resume_monitor_stop:
            for backend in self.backends.values():
                if backend.scheduling_in_progress:
                    continue
                backend.scheduling_in_progress = True
                try:
                    await self._try_resume_paused(backend)
                finally:
                    backend.scheduling_in_progress = False
            await asyncio.sleep(float(interval_sec))

    @staticmethod
    def _estimate_system_prompt_tokens(payload: Dict[str, Any]) -> int:
        """Estimate system prompt tokens from the first request payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("input_text")
                    if isinstance(text, str):
                        parts.append(text)
        
        if not parts:
            return 0
        
        text = "\n".join(parts)
        return max(0, len(text) // 5)

    def get_or_create_program(self, program_id: str) -> ProgramState:
        """Get existing program or create new one assigned to least loaded backend."""
        if program_id not in self.programs:
            backend = self.select_backend_for_new_program()
            profile = ProfileState(program_id=program_id) if self.profile_enabled else None
            self.programs[program_id] = ProgramState(backend_url=backend.url, profile=profile)
            logger.debug(f"Created program {program_id} on backend {backend.url}")
        return self.programs[program_id]

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get the backend assigned to a program."""
        state = self.programs.get(program_id)
        if state and state.backend_url in self.backends:
            return self.backends[state.backend_url]
        return self.get_default_backend()

    async def update_program_before_request(self, program_id: str, state: ProgramState, payload: Dict[str, Any]) -> bool:
        """Update program state before sending request to vLLM.
        
        If scheduling_enabled=False (default mode): pure proxy, no capacity checks.
        If scheduling_enabled=True (tr mode): full capacity-based scheduling.
        
        Returns: True if can proceed
        """
        state.step_count += 1
        # If we are receiving a request, the previous tool execution must have finished.
        # This is also true for the first step (step_count == 1).
        state.tool_finished = True
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.status = ProgramStatus.REASONING
            return True
        
        is_new_program = state.step_count == 1

        if is_new_program and BackendState.shared_token is None:
            shared_tokens = self._estimate_system_prompt_tokens(payload)
            BackendState.shared_token = shared_tokens
            logger.info(f"Set BackendState.shared_token={shared_tokens} from first request system prompt")
        
        # On first request, estimate total_tokens and add to total tracking
        if is_new_program:
            state.total_tokens = state.context_len // 5
            backend.add_total_program(state.total_tokens)
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
            if state.status == ProgramStatus.ACTING:
                backend.shift_tokens_to_reasoning(state.total_tokens)
            if is_new_program:
                backend.add_active_program(state.total_tokens, is_acting=False)
            state.status = ProgramStatus.REASONING
            return True
        
        # ---------------------------------------------------------------------
        # TR mode: full capacity-based scheduling
        # ---------------------------------------------------------------------
        should_pause = False
        
        # Check 1: If paused by scheduler, wait for resume.
        # Only wait if PAUSED was set by scheduler (new programs start as PAUSED)
        if state.status == ProgramStatus.PAUSED and state.waiting_event is not None:
            # Try to resume immediately (avoid waiting for external events).
            for b in self.backends.values():
                if not b.scheduling_in_progress:
                    b.scheduling_in_progress = True
                    try:
                        await self._try_resume_paused(b)
                    finally:
                        b.scheduling_in_progress = False
            await self._wait_for_resume(program_id, state)
            return True
        
        # Check 2: If marked for pause, pause now
        if state.marked_for_pause:
            logger.info(f"Program {program_id} was marked, pausing now")
            await self._clear_mark_and_pause(program_id, state)
            await self._wait_for_resume(program_id, state)
            return True

        # Check 3: New program fairness (must queue if others waiting)
        has_waiting_programs = self._count_resumable_paused() > 0 or backend.future_paused_tokens > 0
        if is_new_program and has_waiting_programs:
            logger.info(
                "New program %s must wait (paused_global=%s, marked=%s)",
                program_id,
                self._count_resumable_paused(),
                backend.future_paused_tokens,
            )
            await self._pause_new_program(program_id, state, backend)
            await self._wait_for_resume(program_id, state)
            return True
            
        
        # Capacity enforcement (skipped if another scheduling in progress)
        if not backend.scheduling_in_progress:
            backend.scheduling_in_progress = True
            try:
                if self._backend_capacity_overflow(backend, include_future_release=True) > 0:
                    await self._enforce_capacity(backend)

                # If this program got paused during capacity enforcement, do not proceed.
                # This can happen for existing programs (step_count > 1) that were still ACTING when scheduled.
                if state.status == ProgramStatus.PAUSED and state.waiting_event is not None:
                    should_pause = True
                    
                # Final capacity check for new program
                if is_new_program:
                    if not self._backend_has_capacity(backend, extra_tokens=float(state.total_tokens), extra_count=1):
                        await self._pause_new_program(program_id, state, backend)
                        should_pause = True
            finally:
                backend.scheduling_in_progress = False

        
        # Proceed to REASONING
        if not should_pause:
            if is_new_program:
                backend.add_active_program(state.total_tokens, is_acting=False)
            if state.status == ProgramStatus.ACTING:
                state.acting_since = None
                backend.shift_tokens_to_reasoning(state.total_tokens)
            state.status = ProgramStatus.REASONING
        else:
            await self._wait_for_resume(program_id, state)
        
        return True

    async def _pause_new_program(self, program_id: str, state: ProgramState, backend: BackendState) -> None:
        """Pause a new program that hasn't been added to active yet."""
        await self._add_to_paused_pool(program_id, state, backend)
        state.status = ProgramStatus.PAUSED
        
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"New program {program_id} paused (tokens={state.total_tokens})")

    async def update_program_after_request(self, program_id: str, state: ProgramState, total_tokens: int) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts and enforces capacity if needed.
        Uses non-blocking flag: skips enforcement if already in progress.
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        state.tool_finished = False
        state.acting_since = time.time()
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.total_tokens = total_tokens
            return
        
        # Update tokens (always performed)
        old_tokens = state.total_tokens
        state.total_tokens = total_tokens
        backend.shift_tokens_to_acting(old_tokens, total_tokens)
        
        # Capacity enforcement (skipped if already in progress)
        if not backend.scheduling_in_progress:
            backend.scheduling_in_progress = True
            try:
                if self._backend_capacity_overflow(backend, include_future_release=True) > 0:
                    await self._enforce_capacity(backend)
            finally:
                backend.scheduling_in_progress = False

    async def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking and tries to resume paused programs.
        Resumes paused programs based on global pool ordering and capacity.
        Uses non-blocking flag: skips resume if scheduling already in progress.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url)

        def _kick_resume_task(b: BackendState) -> None:
            # Keep /programs/release fast: schedule resume work in background.
            if not self.scheduling_enabled:
                return
            if b.scheduling_in_progress:
                return
            b.scheduling_in_progress = True

            async def _run() -> None:
                try:
                    await self._try_resume_paused(b)
                finally:
                    b.scheduling_in_progress = False

            asyncio.create_task(_run())
        
        if backend:
            # Clean up based on current status (always performed)
            if state.status in (ProgramStatus.REASONING, ProgramStatus.ACTING):
                backend.remove_active_program(
                    state.total_tokens,
                    is_acting=(state.status == ProgramStatus.ACTING),
                )
            elif state.status == ProgramStatus.PAUSED:
                await self._remove_from_paused_pool(program_id)
                if state.waiting_event:
                    state.waiting_event.set()
            
            # Clear mark if was marked
            if state.marked_for_pause:
                backend.future_paused_tokens -= state.total_tokens
                if backend.future_paused_tokens < 0:
                    backend.future_paused_tokens = 0
                state.marked_for_pause = False
            
            # Remove from total tracking
            if state.status != ProgramStatus.STOPPED:
                backend.remove_total_program(state.total_tokens)
            # Best-effort: kick background resume (monitor checks frequently).
            _kick_resume_task(backend)
        
        state.status = ProgramStatus.STOPPED
        
        # Remove from programs dict (no need to keep STOPPED programs)
        del self.programs[program_id]
        
        logger.info(f"Released and removed program: {program_id}")
        return True

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, ProgramState]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    # -------------------------------------------------------------------------
    # Global Paused Pool
    # -------------------------------------------------------------------------

    async def _add_to_paused_pool(self, program_id: str, state: ProgramState, backend: BackendState) -> None:
        """Add or update a program in the global paused pool."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url,
            step_count=state.step_count,
        )
        async with self.pause_resume_lock:
            self.paused_pool[program_id] = paused_info

    async def _remove_from_paused_pool(self, program_id: str) -> Optional[PausedInfo]:
        """Remove a program from the global paused pool."""
        async with self.pause_resume_lock:
            return self.paused_pool.pop(program_id, None)

    def _get_paused_programs_sorted(
        self, ascending: bool = True
    ) -> List[Tuple[str, ProgramState, PausedInfo]]:
        """Get paused programs from the global pool, sorted by total_tokens.

        Callers that require consistency should hold pause_resume_lock.
        """
        programs: List[Tuple[str, ProgramState, PausedInfo]] = []
        for pid, info in self.paused_pool.items():
            state = self.programs.get(pid)
            if not state:
                continue
            programs.append((pid, state, info))
        return sorted(programs, key=lambda x: x[2].total_tokens, reverse=not ascending)

    def get_paused_counts_by_backend(self) -> Dict[str, int]:
        """Count paused programs per backend using paused pool metadata."""
        counts = {url: 0 for url in self.backends}
        for info in self.paused_pool.values():
            if info.origin_backend in counts:
                counts[info.origin_backend] += 1
        return counts

    # -------------------------------------------------------------------------
    # Capacity-based Scheduling (pause/resume)
    # -------------------------------------------------------------------------

    def _get_acting_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, ProgramState]]:
        """Get ACTING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.ACTING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _get_reasoning_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, ProgramState]]:
        """Get REASONING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url 
            and state.status == ProgramStatus.REASONING
            and not state.marked_for_pause  # Exclude already marked
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    async def _pause_program(self, program_id: str, state: ProgramState) -> None:
        """Pause a program: remove from active, add to global paused pool.
        
        Only call this for ACTING programs. REASONING programs should be marked instead.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Remove from active counts
        backend.remove_active_program(
            state.total_tokens,
            is_acting=(state.status == ProgramStatus.ACTING),
        )
        
        # Add to global paused pool.
        await self._add_to_paused_pool(program_id, state, backend)
        state.status = ProgramStatus.PAUSED
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()

        effective_tokens, active_count, _r, _a = self._effective_backend_load(backend.url)
        logger.info(
            "Paused program %s (tokens=%s, effective_active_tokens=%.2f, active_count=%s)",
            program_id,
            state.total_tokens,
            effective_tokens,
            active_count,
        )

        # After pausing one program, attempt to resume (if there is slack).
        await self._try_resume_paused(backend)

    def _mark_program_for_pause(self, program_id: str, state: ProgramState) -> None:
        """Mark a REASONING program for pause. It will be paused on next request.
        
        Adds the program's tokens to future_paused_tokens for capacity calculation.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        state.marked_for_pause = True
        backend.future_paused_tokens += state.total_tokens
        
        logger.info(f"Marked program {program_id} for pause (tokens={state.total_tokens}, future_paused={backend.future_paused_tokens})")

    async def _clear_mark_and_pause(self, program_id: str, state: ProgramState) -> None:
        """Clear the mark from a program and pause it.
        
        Called when a marked program's next request arrives.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Clear the mark and subtract from future_paused_tokens
        state.marked_for_pause = False
        backend.future_paused_tokens -= state.total_tokens
        if backend.future_paused_tokens < 0:
            backend.future_paused_tokens = 0
        
        # Now actually pause the program
        await self._pause_program(program_id, state)

    def _resume_program(
        self,
        program_id: str,
        state: ProgramState,
        target_backend: Optional[BackendState] = None,
    ) -> None:
        """Resume a paused program after it has been claimed from the pool."""
        origin_backend = self.backends.get(state.backend_url)
        backend = target_backend or origin_backend
        if not backend:
            return

        if state.status != ProgramStatus.PAUSED:
            return

        # Allow backend migration on resume and keep per-backend totals consistent.
        if state.backend_url != backend.url:
            if origin_backend:
                origin_backend.remove_total_program(state.total_tokens)
            backend.add_total_program(state.total_tokens)
            state.backend_url = backend.url

        # Add to active counts.
        backend.add_active_program(state.total_tokens, is_acting=False)
        state.status = ProgramStatus.REASONING
        state.acting_since = None
        
        # Signal waiting event
        if state.waiting_event:
            state.waiting_event.set()

        effective_tokens, active_count, _r, _a = self._effective_backend_load(backend.url)
        logger.info(
            "Resumed program %s (tokens=%s, effective_active_tokens=%.2f, active_count=%s)",
            program_id,
            state.total_tokens,
            effective_tokens,
            active_count,
        )

    async def _enforce_capacity(self, backend: BackendState) -> Tuple[int, int, int]:
        """Enforce capacity constraint by pausing ACTING and marking REASONING programs.
        
        Overflow check uses the effective active token model:
          effective_active_tokens = reasoning_tokens + Σ(2^(-t_i) * acting_tokens_i)
        with the same decode buffer and shared-token reuse.
        
        Algorithm:
        1. Pause ACTING programs (smallest first) until no overflow
        2. If still overflow (no more ACTING), mark REASONING programs (smallest first)
        3. After actual pauses, try to backfill with small paused programs to reduce slack.
        
        Note: Respects pause cooldown to prevent rapid oscillation.
              Marking REASONING programs is not subject to cooldown (deferred pause).
        
        Returns: (paused_count, marked_count, resumed_count)
        """
        paused = 0
        marked = 0
        resumed = 0
        
        # Step 1: Pause ACTING programs (smallest first)
        # Respect cooldown: only pause if enough time has passed
        while self._backend_capacity_overflow(backend, include_future_release=True) > 0:
            # Check cooldown before pausing
            if paused > 0 and not backend.can_pause():
                logger.debug("Pause cooldown active, deferring further pauses")
                break
            
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if not acting_programs:
                break  # No more ACTING programs, move to Step 2
            
            program_id, state = acting_programs[0]
            await self._pause_program(program_id, state)
            backend.record_pause()
            paused += 1
        
        # Step 2: Mark REASONING programs if still over capacity
        # Marking is not subject to cooldown (actual pause will happen later)
        while self._backend_capacity_overflow(backend, include_future_release=True) > 0:
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if not reasoning_programs:
                break  # No more REASONING programs to mark
            
            program_id, state = reasoning_programs[0]
            self._mark_program_for_pause(program_id, state)
            marked += 1

        return paused, marked, resumed

    async def _claim_paused_for_backend(
        self, backend: BackendState
    ) -> Optional[Tuple[str, ProgramState, PausedInfo]]:
        """Claim a paused program for a backend in a lock-protected way."""
        async with self.pause_resume_lock:
            paused_programs = [p for p in self._get_paused_programs_sorted(ascending=True) if p[1].tool_finished]
            paused_programs = (
                [p for p in paused_programs if p[1].step_count > 1]
                + [p for p in paused_programs if p[1].step_count <= 1]
            )

            for program_id, state, info in paused_programs:
                if program_id not in self.programs or state.status != ProgramStatus.PAUSED:
                    self.paused_pool.pop(program_id, None)
                    continue
                if self._backend_has_capacity(backend, extra_tokens=float(state.total_tokens), extra_count=1):
                    # Pop is the claim: only one backend can resume this program.
                    self.paused_pool.pop(program_id, None)
                    return program_id, state, info

        return None

    async def _claim_specific_paused(
        self, program_id: str
    ) -> Optional[Tuple[str, ProgramState, PausedInfo]]:
        """Claim a specific paused program in a lock-protected way."""
        async with self.pause_resume_lock:
            info = self.paused_pool.get(program_id)
            if info is None:
                return None
            state = self.programs.get(program_id)
            if not state or state.status != ProgramStatus.PAUSED:
                self.paused_pool.pop(program_id, None)
                return None
            self.paused_pool.pop(program_id, None)
            return program_id, state, info

    async def _try_resume_paused(self, backend: BackendState) -> int:
        """Resume paused programs that fit within capacity.
        
        Resumes smallest programs first from the global pool, prioritizing
        non-first-step programs.
        Respects resume cooldown to prevent rapid oscillation.
        
        Returns: number of programs resumed
        """
        resumed = 0
        
        # Check cooldown before any resume
        if not backend.can_resume():
            logger.debug("Resume cooldown active, deferring resume")
            return 0
        
        while True:
            claim = await self._claim_paused_for_backend(backend)
            if claim is None:
                break
            program_id, state, _info = claim
            self._resume_program(program_id, state, backend)
            backend.record_resume()
            resumed += 1
            # Continue trying to backfill until no more candidates fit.
        
        return resumed

    async def _wait_for_resume(self, program_id: str, state: ProgramState, timeout: float = 1800.0) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (30 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        
        try:
            await asyncio.wait_for(state.waiting_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Program {program_id} wait timeout after {timeout}s, forcing resume")
            claim = await self._claim_specific_paused(program_id)
            if claim is None:
                return
            program_id, state, _info = claim
            self._resume_program(program_id, state)

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = len(self.paused_pool)
        paused_resumable = self._count_resumable_paused()
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        effective_by_backend = self.get_effective_backend_stats()
        paused_counts = self.get_paused_counts_by_backend()
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": paused_counts.get(url, 0),
                "paused_resumable": sum(
                    1 for p in progs.values() if p.status == ProgramStatus.PAUSED and p.tool_finished
                ),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
                "effective": effective_by_backend.get(url, {}),
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
            "paused_resumable": paused_resumable,
            "marked_for_pause": marked,
            "per_backend": per_backend,
        }

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
        """Extract total_tokens from the response."""
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    @staticmethod
    def extract_usage_info(payload: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract usage info from the response.
        
        Returns:
            (total_tokens, prompt_tokens, cached_tokens)
            - cached_tokens is 0 if prompt_tokens_details is null or missing
            - All None if usage is not available
        """
        if not isinstance(payload, dict):
            return None, None, None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None, None, None
        
        total_tokens = None
        prompt_tokens = None
        cached_tokens = 0  # Default to 0 if not available
        
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                total_tokens = int(val)
        
        if "prompt_tokens" in usage:
            val = usage.get("prompt_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                prompt_tokens = int(val)
        
        # Extract cached_tokens from prompt_tokens_details
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            ct = prompt_details.get("cached_tokens")
            if isinstance(ct, (int, float)) and math.isfinite(ct):
                cached_tokens = int(ct)
        
        return total_tokens, prompt_tokens, cached_tokens

    @staticmethod
    def filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        """Filter out hop-by-hop headers."""
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    @staticmethod
    def remove_program_id(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Remove program_id from payload before forwarding to vLLM."""
        payload = payload.copy()
        payload.pop("program_id", None)
        if "extra_body" in payload and isinstance(payload["extra_body"], dict):
            payload["extra_body"] = payload["extra_body"].copy()
            payload["extra_body"].pop("program_id", None)
            if not payload["extra_body"]:
                del payload["extra_body"]
        return payload

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        on_first_token: Callable[[], None] | None = None,
        on_token: Callable[[], None] | None = None,
    ) -> Response:
        """Proxy request to a specific backend.
        
        Args:
            on_usage: Callback with (total_tokens, prompt_tokens, cached_tokens)
        """
        url = backend.completions_url
        
        # Remove program_id before forwarding
        payload = self.remove_program_id(payload)

        if payload.get("stream"):
            # Add stream_options to get usage info in streaming response
            if on_usage is not None:
                stream_options = payload.get("stream_options")
                if stream_options is None:
                    payload["stream_options"] = {"include_usage": True}
                elif isinstance(stream_options, dict):
                    stream_options.setdefault("include_usage", True)

            resp_cm = self.client.stream("POST", url, json=payload)
            resp = await resp_cm.__aenter__()
            headers = self.filtered_headers(resp.headers)
            status = resp.status_code
            media_type = resp.headers.get("content-type")

            async def iterator():
                buffer = b""
                usage_extracted = False
                total_tokens: Optional[int] = None
                prompt_tokens: Optional[int] = None
                cached_tokens: int = 0
                first_token_seen = False
                try:
                    async for chunk in resp.aiter_raw():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            for line in event.split(b"\n"):
                                if not line.startswith(b"data:"):
                                    continue
                                data = line[5:].strip()
                                if not data or data == b"[DONE]":
                                    continue
                                
                                # Profile: track token timing
                                if not first_token_seen:
                                    first_token_seen = True
                                    if on_first_token is not None:
                                        on_first_token()
                                if on_token is not None:
                                    on_token()
                                
                                if usage_extracted:
                                    continue
                                try:
                                    payload_obj = json.loads(data)
                                except Exception:
                                    continue
                                tt, pt, ct = self.extract_usage_info(payload_obj)
                                if tt is not None:
                                    total_tokens = tt
                                    prompt_tokens = pt
                                    cached_tokens = ct
                                    usage_extracted = True
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)
                    if total_tokens is not None and on_usage is not None:
                        await on_usage(total_tokens, prompt_tokens or 0, cached_tokens)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        # Non-streaming request
        resp = await self.client.post(url, json=payload)
        total_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        cached_tokens: int = 0
        try:
            payload_obj = resp.json()
        except Exception:
            payload_obj = None
        tt, pt, ct = self.extract_usage_info(payload_obj)
        if tt is not None:
            total_tokens = tt
            prompt_tokens = pt
            cached_tokens = ct
        if total_tokens is not None and on_usage is not None:
            await on_usage(total_tokens, prompt_tokens or 0, cached_tokens)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self.filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )

    async def proxy_get(self, backend_url: str, path: str) -> Response:
        """Proxy a GET request to a backend."""
        url = f"{backend_url.rstrip('/')}{path}"
        resp = await self.client.get(url)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self.filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
