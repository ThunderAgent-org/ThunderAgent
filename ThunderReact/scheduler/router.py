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

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
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

    def update_program_before_request(self, state: ProgramState, payload: Dict[str, Any]) -> None:
        """Update program state before sending request to vLLM.
        
        Sets status to REASONING (on GPU, running inference).
        Updates backend's active_tokens on first request.
        """
        state.status = ProgramStatus.REASONING
        state.step_count += 1
        
        # context_len = string length of payload
        # pre assign total_tokens for step1 program
        import json
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        
        # On first request, estimate total_tokens and add to backend
        if state.step_count == 1:
            state.total_tokens = state.context_len // 5
            # Add estimated tokens to backend
            backend = self.backends.get(state.backend_url)
            if backend:
                backend.add_program_tokens(state.total_tokens)

    def update_program_after_request(self, state: ProgramState, total_tokens: int) -> None:
        """Update program state after receiving response from vLLM.
        
        Sets status to ACTING (off GPU, executing tool).
        Updates backend's active_tokens with actual value.
        """
        state.status = ProgramStatus.ACTING
        
        # Update backend's active_tokens (from estimate/old value to actual)
        old_tokens = state.total_tokens
        state.total_tokens = total_tokens
        backend = self.backends.get(state.backend_url)
        if backend:
            backend.update_program_tokens(old_tokens, total_tokens)

    def release_program(self, program_id: str) -> bool:
        """Stop a program (keep in records with stopped status).
        
        Removes program's tokens from backend's active_tokens.
        Only removes tokens if program was active (REASONING/ACTING).
        """
        if program_id in self.programs:
            state = self.programs[program_id]
            # Only remove tokens if program was active (not already STOPPED/PAUSED)
            if state.status in (ProgramStatus.REASONING, ProgramStatus.ACTING):
                backend = self.backends.get(state.backend_url)
                if backend:
                    backend.remove_program_tokens(state.total_tokens)
            state.status = ProgramStatus.STOPPED
            logger.info(f"Released program: {program_id}")
            return True
        return False

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, ProgramState]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = sum(1 for p in self.programs.values() if p.status == ProgramStatus.PAUSED)
        
        # Per-backend stats
        per_backend = {}
        for url in self.backends:
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": sum(1 for p in progs.values() if p.status == ProgramStatus.PAUSED),
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
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
