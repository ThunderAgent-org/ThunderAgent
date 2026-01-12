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

# Interval for periodic capacity check (seconds)
CAPACITY_CHECK_INTERVAL = 2.0


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    def __init__(self, backend_urls: str | List[str], *, profile_enabled: bool = False) -> None:
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

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        
        # Capacity check task
        self._capacity_check_task: Optional[asyncio.Task] = None

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
        
        # Start capacity check task
        self._capacity_check_task = asyncio.create_task(self._capacity_check_loop())

    async def stop(self):
        """Stop the router."""
        # Stop capacity check task
        if self._capacity_check_task:
            self._capacity_check_task.cancel()
            try:
                await self._capacity_check_task
            except asyncio.CancelledError:
                pass
            self._capacity_check_task = None
        
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
        
        Checks capacity and may pause the program if over capacity.
        Sets status to REASONING (on GPU, running inference).
        
        Returns: True if can proceed, False if should not proceed (shouldn't happen normally)
        """
        state.step_count += 1
        
        # context_len = string length of payload
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.status = ProgramStatus.REASONING
            return True
        
        # On first request, estimate total_tokens
        is_first_request = state.step_count == 1
        if is_first_request:
            state.total_tokens = state.context_len // 5
            # Add to total_program_tokens (always track)
            backend.add_total_program(state.total_tokens)
        
        # Check capacity
        if state.status == ProgramStatus.PAUSED:
            # Already paused, need to wait for resume
            await self._wait_for_resume(program_id, state)
        elif is_first_request:
            # New program: check if capacity is blocked or no capacity
            if backend.capacity_blocked or not backend.has_capacity(extra_tokens=state.total_tokens, extra_count=1):
                # No capacity or blocked: pause this program and wait
                self._pause_program(program_id, state)
                await self._wait_for_resume(program_id, state)
            else:
                # Has capacity: add to active
                backend.add_active_program(state.total_tokens)
                state.status = ProgramStatus.REASONING
        else:
            # Existing program coming back from ACTING
            # Check if capacity is blocked - if so, pause and wait
            if backend.capacity_blocked:
                self._pause_program(program_id, state)
                await self._wait_for_resume(program_id, state)
            else:
                # It's already counted in active, just set status
                state.status = ProgramStatus.REASONING
        
        return True

    def update_program_after_request(self, program_id: str, state: ProgramState, total_tokens: int) -> None:
        """Update program state after receiving response from vLLM.
        
        Sets status to ACTING (off GPU, executing tool).
        Updates backend's tokens and checks capacity:
        - If program was marked_for_pause: pause it immediately
        - If over capacity: pause smallest ACTING programs, mark REASONING programs
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        
        backend = self.backends.get(state.backend_url)
        if not backend:
            state.total_tokens = total_tokens
            return
        
        # Update tokens (both active and total)
        old_tokens = state.total_tokens
        state.total_tokens = total_tokens
        backend.update_program_tokens(old_tokens, total_tokens)
        
        # Check if this program was marked for pause
        if state.marked_for_pause:
            logger.info(f"Program {program_id} was marked for pause, pausing now")
            self._pause_program(program_id, state)
            return
        
        # Check capacity after update
        if backend.capacity_overflow() > 0:
            # Over capacity: pause/mark programs
            paused_count, marked_count = self._try_pause_overflowing(backend)
            if paused_count > 0 or marked_count > 0:
                logger.info(f"Capacity overflow: paused={paused_count}, marked={marked_count}")

    def release_program(self, program_id: str) -> bool:
        """Stop a program (keep in records with stopped status).
        
        Removes program's tokens from all tracking.
        Tries to resume paused programs if capacity becomes available.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url)
        
        if backend:
            # Remove from active if was REASONING/ACTING
            if state.status in (ProgramStatus.REASONING, ProgramStatus.ACTING):
                backend.remove_active_program(state.total_tokens)
            elif state.status == ProgramStatus.PAUSED:
                # Remove from paused set
                backend.paused_programs.discard(program_id)
                # Signal waiting event to unblock any waiters
                if state.waiting_event:
                    state.waiting_event.set()
            
            # Clear marked_for_pause flag
            state.marked_for_pause = False
            backend.marked_for_pause.discard(program_id)
            
            # Always remove from total (unless already STOPPED)
            if state.status != ProgramStatus.STOPPED:
                backend.remove_total_program(state.total_tokens)
            
            # Try to unblock capacity
            self._try_unblock_capacity(backend)
            
            # Try to resume paused programs
            resumed_count = self._try_resume_paused(backend)
            if resumed_count > 0:
                logger.info(f"Resumed {resumed_count} paused programs after release")
        
        state.status = ProgramStatus.STOPPED
        logger.info(f"Released program: {program_id}")
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

    def _get_acting_programs_sorted(self, backend_url: str) -> List[Tuple[str, ProgramState]]:
        """Get ACTING programs on a backend, sorted by total_tokens (smallest first)."""
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.ACTING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens)

    def _get_reasoning_programs_sorted(self, backend_url: str) -> List[Tuple[str, ProgramState]]:
        """Get REASONING programs on a backend, sorted by total_tokens (smallest first)."""
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.REASONING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens)

    def _get_paused_programs_sorted(self, backend_url: str) -> List[Tuple[str, ProgramState]]:
        """Get PAUSED programs on a backend, sorted by total_tokens (smallest first)."""
        backend = self.backends.get(backend_url)
        if not backend:
            return []
        programs = [
            (pid, self.programs[pid]) for pid in backend.paused_programs
            if pid in self.programs
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens)

    def _pause_program(self, program_id: str, state: ProgramState) -> None:
        """Pause a program: remove from active, add to paused set."""
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Remove from active counts (only if was active)
        if state.status in (ProgramStatus.REASONING, ProgramStatus.ACTING):
            backend.remove_active_program(state.total_tokens)
        
        # Add to paused set
        backend.paused_programs.add(program_id)
        state.status = ProgramStatus.PAUSED
        
        # Clear marked_for_pause flag (if was marked)
        state.marked_for_pause = False
        backend.marked_for_pause.discard(program_id)
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"Paused program {program_id} (tokens={state.total_tokens}, active_tokens={backend.active_program_tokens})")

    def _resume_program(self, program_id: str, state: ProgramState) -> None:
        """Resume a program: add to active, remove from paused set."""
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Add to active counts
        backend.add_active_program(state.total_tokens)
        
        # Remove from paused set
        backend.paused_programs.discard(program_id)
        state.status = ProgramStatus.REASONING
        
        # Signal waiting event
        if state.waiting_event:
            state.waiting_event.set()
        
        logger.info(f"Resumed program {program_id} (tokens={state.total_tokens}, active_tokens={backend.active_program_tokens})")

    def _try_resume_paused(self, backend: BackendState) -> int:
        """Try to resume paused programs that fit within capacity.
        
        Returns: number of programs resumed
        """
        resumed = 0
        paused_programs = self._get_paused_programs_sorted(backend.url)
        
        for program_id, state in paused_programs:
            if backend.has_capacity(extra_tokens=state.total_tokens, extra_count=1):
                self._resume_program(program_id, state)
                resumed += 1
            else:
                break  # Smallest doesn't fit, stop trying
        
        return resumed

    def _mark_program_for_pause(self, program_id: str, state: ProgramState) -> None:
        """Mark a REASONING program to be paused when it becomes ACTING."""
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        state.marked_for_pause = True
        backend.marked_for_pause.add(program_id)
        logger.info(f"Marked program {program_id} for pause (tokens={state.total_tokens})")

    def _try_pause_overflowing(self, backend: BackendState) -> Tuple[int, int]:
        """Pause ACTING programs and mark REASONING programs until capacity is satisfied.
        
        Algorithm:
        1. First, pause all ACTING programs (smallest first) - immediate effect
        2. If still over capacity, mark REASONING programs (smallest first)
           - They will be paused when transitioning to ACTING
        3. Set capacity_blocked = True to block new requests
        
        Returns: (paused_count, marked_count)
        """
        paused = 0
        marked = 0
        
        # Step 1: Pause ACTING programs (smallest first)
        while backend.capacity_overflow() > 0:
            acting_programs = self._get_acting_programs_sorted(backend.url)
            if not acting_programs:
                break  # No more ACTING programs to pause
            
            program_id, state = acting_programs[0]
            self._pause_program(program_id, state)
            paused += 1
        
        # Step 2: Mark REASONING programs if still over capacity
        while backend.capacity_overflow() > 0:
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url)
            # Filter out already marked ones
            reasoning_programs = [(pid, s) for pid, s in reasoning_programs if not s.marked_for_pause]
            
            if not reasoning_programs:
                break  # No more REASONING programs to mark
            
            program_id, state = reasoning_programs[0]
            self._mark_program_for_pause(program_id, state)
            marked += 1
            # Note: marked programs are still counted in active_tokens until they transition to ACTING
        
        # Step 3: Set capacity_blocked if still over or have marked programs
        if backend.capacity_overflow() > 0 or len(backend.marked_for_pause) > 0:
            if not backend.capacity_blocked:
                backend.capacity_blocked = True
                backend.capacity_unblocked_event.clear()
                logger.info(f"Backend {backend.url} capacity blocked (overflow={backend.capacity_overflow()}, marked={len(backend.marked_for_pause)})")
        
        return paused, marked
    
    def _try_unblock_capacity(self, backend: BackendState) -> bool:
        """Try to unblock capacity if conditions are met.
        
        Conditions for unblocking:
        - No capacity overflow
        - No programs marked for pause
        
        Returns: True if unblocked
        """
        if not backend.capacity_blocked:
            return False
        
        if backend.capacity_overflow() <= 0 and len(backend.marked_for_pause) == 0:
            backend.capacity_blocked = False
            backend.capacity_unblocked_event.set()
            logger.info(f"Backend {backend.url} capacity unblocked")
            return True
        
        return False

    async def _capacity_check_loop(self) -> None:
        """Periodic capacity check loop (runs every CAPACITY_CHECK_INTERVAL seconds).
        
        For each backend:
        - If over capacity: pause ACTING programs, mark REASONING programs, block new requests
        - If under capacity and no marked: unblock and try resume paused programs
        """
        logger.info(f"Capacity check loop started (interval={CAPACITY_CHECK_INTERVAL}s)")
        
        while True:
            try:
                await asyncio.sleep(CAPACITY_CHECK_INTERVAL)
                
                for backend in self.backends.values():
                    if not backend.cache_config:
                        continue  # No config, skip
                    
                    overflow = backend.capacity_overflow()
                    
                    if overflow > 0:
                        # Over capacity: pause/mark
                        paused, marked = self._try_pause_overflowing(backend)
                        if paused > 0 or marked > 0:
                            logger.info(f"Capacity check: paused={paused}, marked={marked}, overflow={overflow}")
                    else:
                        # Under capacity: try to unblock and resume
                        self._try_unblock_capacity(backend)
                        
                        # Try to resume paused programs if not blocked
                        if not backend.capacity_blocked:
                            resumed = self._try_resume_paused(backend)
                            if resumed > 0:
                                logger.info(f"Capacity check: resumed {resumed} paused programs")
                                
            except asyncio.CancelledError:
                logger.info("Capacity check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Capacity check loop error: {e}")

    async def _wait_for_resume(self, program_id: str, state: ProgramState, timeout: float = 1800.0) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (20 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        
        try:
            await asyncio.wait_for(state.waiting_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Program {program_id} wait timeout after {timeout}s, forcing resume")
            # Force resume: add back to active (even if over capacity)
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
                "marked_for_pause": len(backend.marked_for_pause),
                "capacity_blocked": backend.capacity_blocked,
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
