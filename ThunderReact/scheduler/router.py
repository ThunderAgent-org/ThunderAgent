"""Router with program state tracking - supports multiple backends."""
import asyncio
import json
import logging
import math
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple

import httpx
from fastapi.responses import Response, StreamingResponse

from ..backend import BackendState
from ..program import ProgramState, ProgramStatus
from ..profile.state import ProfileState
from ..config import get_config

logger = logging.getLogger(__name__)


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
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )

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

    async def stop(self):
        """Stop the router."""
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
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.status = ProgramStatus.REASONING
            return True

        if state.status == ProgramStatus.ACTING:
            backend.shift_tokens_to_reasoning(state.total_tokens)
        
        is_new_program = state.step_count == 1

        if is_new_program and BackendState.shared_token is None:
            shared_tokens = self._estimate_system_prompt_tokens(payload)
            BackendState.shared_token = shared_tokens
            logger.info(f"Set SHARED_TOKEN={shared_tokens} from first request system prompt")
        
        # On first request, estimate total_tokens and add to total tracking
        if is_new_program:
            state.total_tokens = state.context_len // 5
            backend.add_total_program(state.total_tokens)
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
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
            await self._wait_for_resume(program_id, state)
            return True
        
        # Check 2: If marked for pause, pause now
        if state.marked_for_pause:
            logger.info(f"Program {program_id} was marked, pausing now")
            self._clear_mark_and_pause(program_id, state)
            await self._wait_for_resume(program_id, state)
            return True
        
        # Check 3: New program fairness (must queue if others waiting)
        has_waiting_programs = len(backend.paused_programs) > 0 or backend.future_paused_tokens > 0
        if is_new_program and has_waiting_programs:
            logger.info(f"New program {program_id} must wait (paused={len(backend.paused_programs)}, marked={backend.future_paused_tokens})")
            self._pause_new_program(program_id, state, backend)
            await self._wait_for_resume(program_id, state)
            return True
        
        # Capacity enforcement (skipped if another scheduling in progress)
        if not backend.scheduling_in_progress:
            backend.scheduling_in_progress = True
            try:
                if backend.capacity_overflow(include_future_release=True) > 0:
                    self._enforce_capacity(backend)
                
                # Final capacity check for new program
                if is_new_program:
                    if not backend.has_capacity(extra_tokens=state.total_tokens, extra_count=1):
                        self._pause_new_program(program_id, state, backend)
                        should_pause = True
            finally:
                backend.scheduling_in_progress = False
        
        # Proceed to REASONING
        if not should_pause:
            if is_new_program:
                backend.add_active_program(state.total_tokens, is_acting=False)
            state.status = ProgramStatus.REASONING
        else:
            await self._wait_for_resume(program_id, state)
        
        return True

    def _pause_new_program(self, program_id: str, state: ProgramState, backend: BackendState) -> None:
        """Pause a new program that hasn't been added to active yet."""
        backend.paused_programs.add(program_id)
        state.status = ProgramStatus.PAUSED
        
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"New program {program_id} paused (tokens={state.total_tokens})")

    def update_program_after_request(self, program_id: str, state: ProgramState, total_tokens: int) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts and enforces capacity if needed.
        Uses non-blocking flag: skips enforcement if already in progress.
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        
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
                if backend.capacity_overflow(include_future_release=True) > 0:
                    self._enforce_capacity(backend)
            finally:
                backend.scheduling_in_progress = False

    def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking and tries to resume paused programs.
        Resumes largest paused programs first.
        Uses non-blocking flag: skips resume if scheduling already in progress.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url)
        
        if backend:
            # Clean up based on current status (always performed)
            if state.status in (ProgramStatus.REASONING, ProgramStatus.ACTING):
                backend.remove_active_program(
                    state.total_tokens,
                    is_acting=(state.status == ProgramStatus.ACTING),
                )
            elif state.status == ProgramStatus.PAUSED:
                backend.paused_programs.discard(program_id)
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
            
            # Resume paused programs (skipped if scheduling in progress)
            if not backend.scheduling_in_progress:
                backend.scheduling_in_progress = True
                try:
                    resumed = self._try_resume_paused(backend)
                    if resumed > 0:
                        logger.info(f"Resumed {resumed} paused programs after release")
                finally:
                    backend.scheduling_in_progress = False
        
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

    def _get_paused_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, ProgramState]]:
        """Get PAUSED programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        backend = self.backends.get(backend_url)
        if not backend:
            return []
        programs = [
            (pid, self.programs[pid]) for pid in backend.paused_programs
            if pid in self.programs
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _pause_program(self, program_id: str, state: ProgramState) -> None:
        """Pause a program: remove from active, add to paused set.
        
        Only call this for ACTING programs. REASONING programs should be marked instead.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Remove from active counts
        backend.remove_active_program(state.total_tokens, is_acting=True)
        
        # Add to paused set
        backend.paused_programs.add(program_id)
        state.status = ProgramStatus.PAUSED
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"Paused program {program_id} (tokens={state.total_tokens}, active={backend.active_program_tokens})")

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

    def _clear_mark_and_pause(self, program_id: str, state: ProgramState) -> None:
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
        self._pause_program(program_id, state)

    def _resume_program(self, program_id: str, state: ProgramState) -> None:
        """Resume a paused program: add to active, remove from paused set."""
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Add to active counts
        backend.add_active_program(state.total_tokens, is_acting=False)
        
        # Remove from paused set
        backend.paused_programs.discard(program_id)
        state.status = ProgramStatus.REASONING
        
        # Signal waiting event
        if state.waiting_event:
            state.waiting_event.set()
        
        logger.info(f"Resumed program {program_id} (tokens={state.total_tokens}, active={backend.active_program_tokens})")

    def _enforce_capacity(self, backend: BackendState) -> Tuple[int, int, int]:
        """Enforce capacity constraint by pausing ACTING and marking REASONING programs.
        
        Overflow check: active_tokens - future_paused_tokens + buffer > total_tokens
        
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
        while backend.capacity_overflow(include_future_release=True) > 0:
            # Check cooldown before pausing
            if paused > 0 and not backend.can_pause():
                logger.debug("Pause cooldown active, deferring further pauses")
                break
            
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if not acting_programs:
                break  # No more ACTING programs, move to Step 2
            
            program_id, state = acting_programs[0]
            self._pause_program(program_id, state)
            backend.record_pause()
            paused += 1
        
        # Step 2: Mark REASONING programs if still over capacity
        # Marking is not subject to cooldown (actual pause will happen later)
        while backend.capacity_overflow(include_future_release=True) > 0:
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if not reasoning_programs:
                break  # No more REASONING programs to mark
            
            program_id, state = reasoning_programs[0]
            self._mark_program_for_pause(program_id, state)
            marked += 1

        # Step 3：After actual pauses, try to backfill with small paused programs to reduce slack.
        if paused > 0:
            resumed = self._try_resume_paused(backend)
            if resumed > 0:
                logger.info(f"Resumed {resumed} paused programs after capacity enforcement")
        
        if paused > 0 or marked > 0:
            logger.info(f"Capacity enforcement: paused={paused}, marked={marked}, "
                       f"active={backend.active_program_tokens}, future_paused={backend.future_paused_tokens}")
        
        return paused, marked, resumed

    def _try_resume_paused(self, backend: BackendState) -> int:
        """Resume paused programs that fit within capacity.
        
        Resumes smallest programs first, prioritizing non-first-step programs.
        Respects resume cooldown to prevent rapid oscillation.
        
        Returns: number of programs resumed
        """
        resumed = 0
        
        # Check cooldown before any resume
        if not backend.can_resume():
            logger.debug("Resume cooldown active, deferring resume")
            return 0
        
        # Get paused programs sorted by tokens (smallest first)
        paused_programs = self._get_paused_programs_sorted(backend.url, ascending=True)
        # Prefer non-first-step programs, then step-1 programs
        paused_programs = (
            [p for p in paused_programs if p[1].step_count > 1]
            + [p for p in paused_programs if p[1].step_count <= 1]
        )
        
        for program_id, state in paused_programs:
            if backend.has_capacity(extra_tokens=state.total_tokens, extra_count=1):
                self._resume_program(program_id, state)
                backend.record_resume()
                resumed += 1
            # Continue checking smaller programs even if this one doesn't fit
        
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
            self._resume_program(program_id, state)

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = sum(1 for p in self.programs.values() if p.status == ProgramStatus.PAUSED)
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": sum(1 for p in progs.values() if p.status == ProgramStatus.PAUSED),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
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
