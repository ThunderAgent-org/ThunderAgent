import asyncio
import hashlib
import json
import math
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Awaitable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families

# Configuration
VLLM_BACKENDS = os.getenv(
    "VLLM_BACKENDS", 
    "http://localhost:8100"
).split(",")

THRASHING_DISTANCE = 0.05
CONTROL_TIME_WINDOW_S = 5.0
CONTROL_STEP = 0.2


PAUSE_TRIGGER_USAGE = 0.95
RESUME_TRIGGER_USAGE = 0.85
TRANSFER_IMBALANCE_TRIGGER = 0.30


KV_CACHE_TOKEN_BUDGET = 788976
OUTPUT_TOKEN_ESTIMATE = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProgramState:
    context_len: int
    step_count: int = 0
    inflight: bool = False
    paused: bool = False
    transfer_target: Optional[str] = None
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    waiting_on_resume: bool = False

    @property
    def est_tokens(self) -> int:
        return self.context_len + OUTPUT_TOKEN_ESTIMATE

    def __post_init__(self):
        if not self.paused:
            self.resume_event.set()


@dataclass
class BackendState:
    url: str
    usage: float = 0.0
    healthy: bool = True
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    programs: Dict[str, ProgramState] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    overload_start_time: float = 0.0
    underload_start_time: float = 0.0

    @property
    def metrics_url(self) -> str:
        return f"{self.url}/metrics"

    @property
    def completions_url(self) -> str:
        return f"{self.url}/v1/chat/completions"

    @property
    def total_inflight(self) -> int:
        return sum(1 for p in self.programs.values() if p.inflight)

    @property
    def paused_count(self) -> int:
        return sum(1 for p in self.programs.values() if p.paused)


class MultiBackendRouter:
    def __init__(self, backend_urls: List[str]) -> None:
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        self.program_affinity: Dict[str, str] = {}  
        self._transfer_imbalance_start_time: float = 0.0
        
        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.metrics_client = httpx.AsyncClient(timeout=5.0)
        self.monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        self.monitor_task = asyncio.create_task(self.monitor_loop())
        logger.info(f"Started router with {len(self.backends)} backends: {list(self.backends.keys())}")

    async def stop(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
        await self.metrics_client.aclose()

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    @staticmethod
    def parse_metric_value(line: str) -> Optional[float]:
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = float(parts[1])
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value

    def extract_metrics_fallback(self, text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        usage = None
        running = None
        waiting = None
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc"):
                if usage is None:
                    usage = self.parse_metric_value(line)
            elif line.startswith("vllm:num_requests_running"):
                if running is None:
                    running = self.parse_metric_value(line)
            elif line.startswith("vllm:num_requests_waiting"):
                if waiting is None:
                    waiting = self.parse_metric_value(line)
            if usage is not None and running is not None and waiting is not None:
                break
        return usage, running, waiting

    async def monitor_loop(self):
        while True:
            tasks = [
                self.fetch_backend_usage(backend)
                for backend in self.backends.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            now = time.time()
            await self.schedule_global_transfers(now)

            for backend in self.backends.values():
                async with backend.lock:
                    self.update_backend_state(backend)

            await asyncio.sleep(0.5)

    async def fetch_backend_usage(self, backend: BackendState):
        try:
            resp = await self.metrics_client.get(backend.metrics_url, timeout=3.0)
            resp.raise_for_status()

            running_req = None
            waiting_req = None
            usage_val = None

            try:
                families = list(text_string_to_metric_families(resp.text))
            except Exception as exc:
                logger.warning(
                    f"Failed to parse metrics from {backend.url}, falling back to line parsing: {exc!r}"
                )
                usage_val, running_req, waiting_req = self.extract_metrics_fallback(resp.text)
            else:
                for family in families:
                    if family.name == "vllm:kv_cache_usage_perc":
                        values = [
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        ]
                        if values:
                            usage_val = max(values)
                    elif family.name == "vllm:num_requests_running":
                        running_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )
                    elif family.name == "vllm:num_requests_waiting":
                        waiting_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )

            if usage_val is not None:
                backend.usage = usage_val / 100.0 if usage_val > 1.0 else usage_val
            if running_req is not None:
                backend.running_requests = running_req
            if waiting_req is not None:
                backend.waiting_requests = waiting_req
            backend.healthy = True
        except Exception as exc:
            logger.warning(f"Failed to fetch metrics from {backend.url}: {exc}")
            backend.healthy = False

    # -------------------------------------------------------------------------
    # Pause/Resume/Transfer Logic 
    # -------------------------------------------------------------------------

    def update_backend_state(self, backend: BackendState):
        now = time.time()
        usage = backend.usage

        thrashing_margin = max(0.0, THRASHING_DISTANCE)
        low_watermark = max(0.0, PAUSE_TRIGGER_USAGE - thrashing_margin)

        if usage >= PAUSE_TRIGGER_USAGE:
            if backend.overload_start_time == 0.0:
                backend.overload_start_time = now
        else:
            backend.overload_start_time = 0.0

        overload_sustained = (
            backend.overload_start_time > 0.0
            and (now - backend.overload_start_time) >= CONTROL_TIME_WINDOW_S
        )
        if overload_sustained:
            tokens_to_pause = (
                max(0.0, usage - low_watermark)
                * KV_CACHE_TOKEN_BUDGET
                * CONTROL_STEP
            )
            self.pause_programs_on_backend(backend, tokens_to_pause)
            return

        if usage < RESUME_TRIGGER_USAGE:
            if backend.underload_start_time == 0.0:
                backend.underload_start_time = now
        else:
            backend.underload_start_time = 0.0

        underload_sustained = (
            backend.underload_start_time > 0.0
            and (now - backend.underload_start_time) >= CONTROL_TIME_WINDOW_S
        )
        if underload_sustained:
            high_watermark = min(1.0, RESUME_TRIGGER_USAGE + thrashing_margin)
            tokens_to_resume = (
                max(0.0, high_watermark - usage)
                * KV_CACHE_TOKEN_BUDGET
                * CONTROL_STEP
            )
            self.resume_programs_on_backend(backend, tokens_to_resume)

    def pause_programs_on_backend(
        self,
        backend: BackendState,
        tokens_to_pause: float,
    ):
        """Pause low-priority programs on this backend."""
        active = [
            (pid, state)
            for pid, state in backend.programs.items()
            if not state.paused
            and state.transfer_target is None
        ]
        if not active:
            return

        if tokens_to_pause <= 0:
            return
        active.sort(key=lambda item: item[1].est_tokens)

        paused_tokens = 0
        newly_paused = []

        for pid, state in active:
            if paused_tokens >= tokens_to_pause:
                break

            state.paused = True
            state.resume_event.clear()

            paused_tokens += state.est_tokens
            newly_paused.append(pid)

        if newly_paused:
            logger.info(
                f"[{backend.url}] Paused {len(newly_paused)} programs "
                f"(usage={backend.usage:.2%}, tokens={paused_tokens}): {newly_paused}"
            )

    def resume_programs_on_backend(self, backend: BackendState, tokens_to_resume: float):
        """Resume paused programs on this backend."""
        if tokens_to_resume <= 0:
            return

        paused = [
            (pid, state)
            for pid, state in backend.programs.items()
            if state.paused and state.transfer_target is None
        ]
        if not paused:
            return

        paused.sort(key=lambda item: -item[1].est_tokens)

        resumed_tokens = 0
        resumed_pids = []

        for pid, state in paused:
            if resumed_tokens >= tokens_to_resume:
                break
            state.paused = False
            state.resume_event.set()
            resumed_tokens += state.est_tokens
            resumed_pids.append(pid)

        if resumed_pids:
            logger.info(
                f"[{backend.url}] Resumed {len(resumed_pids)} programs "
                f"(usage={backend.usage:.2%}, budget={tokens_to_resume:.0f}): {resumed_pids}"
            )

    async def schedule_global_transfers(self, now: float) -> None:
        healthy = [b for b in self.backends.values() if b.healthy]
        if len(healthy) < 2:
            self._transfer_imbalance_start_time = 0.0
            return

        source = max(healthy, key=lambda b: b.usage)
        target = min(healthy, key=lambda b: b.usage)
        imbalance = source.usage - target.usage
        if imbalance < TRANSFER_IMBALANCE_TRIGGER:
            self._transfer_imbalance_start_time = 0.0
            return

        if self._transfer_imbalance_start_time == 0.0:
            self._transfer_imbalance_start_time = now
            return
        if (now - self._transfer_imbalance_start_time) < CONTROL_TIME_WINDOW_S:
            return

        transfer_budget = imbalance * KV_CACHE_TOKEN_BUDGET * CONTROL_STEP
        target_free_tokens = max(0.0, 1.0 - target.usage) * KV_CACHE_TOKEN_BUDGET
        transfer_budget = min(transfer_budget, target_free_tokens)
        if transfer_budget <= 0:
            return

        transfer_pids: list[str] = []
        transfer_tokens = 0.0
        async with source.lock:
            active = [
                (pid, state)
                for pid, state in source.programs.items()
                if state.transfer_target is None
            ]
            if not active:
                return

            active.sort(key=lambda item: item[1].est_tokens)
            for pid, state in active:
                if transfer_tokens >= transfer_budget:
                    break
                state.transfer_target = target.url
                if state.paused and state.waiting_on_resume:
                    state.paused = False
                    state.resume_event.set()
                transfer_tokens += state.est_tokens
                transfer_pids.append(pid)

        if transfer_pids:
            self._transfer_imbalance_start_time = 0.0
            logger.info(
                f"[transfer] Scheduled {len(transfer_pids)} programs "
                f"from {source.url} (usage={source.usage:.2%}) "
                f"to {target.url} (usage={target.usage:.2%}) "
                f"(imbalance={imbalance:.2%}, budget={transfer_budget:.0f}, tokens={transfer_tokens:.0f}): "
                f"{transfer_pids}"
            )
            


    async def apply_pending_transfer(
        self,
        program_id: str,
        backend: BackendState,
    ) -> BackendState:
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is None or state.transfer_target is None:
                return backend
            target_url = state.transfer_target

        target_backend = self.backends.get(target_url)
        if target_backend is None or not target_backend.healthy:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state and state.transfer_target == target_url:
                    state.transfer_target = None
            return backend

        if target_backend.url == backend.url:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state:
                    state.transfer_target = None
            return backend

        first, second = sorted([backend, target_backend], key=lambda b: b.url)
        async with first.lock:
            async with second.lock:
                state = backend.programs.get(program_id)
                if (
                    state is None
                    or state.transfer_target != target_url
                    or state.inflight
                ):
                    return backend
                del backend.programs[program_id]
                target_backend.programs[program_id] = state
                self.program_affinity[program_id] = target_url
                state.transfer_target = None
                state.paused = False
                state.resume_event.set()
                return target_backend

    # -------------------------------------------------------------------------
    # Backend Selection (Sticky + Load Balancing)
    # -------------------------------------------------------------------------

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get backend for a program, using sticky routing or least-loaded assignment."""
        # Check existing affinity
        if program_id in self.program_affinity:
            backend_url = self.program_affinity[program_id]
            backend = self.backends.get(backend_url)
            if backend and backend.healthy:
                return backend
            # Backend is unhealthy, need to reassign
            logger.warning(
                f"Backend {backend_url} unhealthy for {program_id}, reassigning"
            )
            del self.program_affinity[program_id]

        # New program: assign to least loaded backend
        if program_id == "default":
            logger.warning("Missing job_id in extra_body; routing will not be balanced across backends.")
        backend = self.pick_least_loaded_backend(program_id)
        self.program_affinity[program_id] = backend.url
        logger.debug(f"Assigned {program_id} to {backend.url}")
        return backend

    def pick_least_loaded_backend(self, program_id: str) -> BackendState:
        """Select a healthy backend, prefer those with no paused programs."""
        healthy = [b for b in self.backends.values() if b.healthy]
        if not healthy:
            logger.warning("No healthy backends, using first available")
            return list(self.backends.values())[0]

        candidates = [b for b in healthy if b.paused_count == 0] or healthy

        def score(b: BackendState) -> float:
            return b.running_requests + b.waiting_requests * 4.0

        scored = [(b, score(b)) for b in candidates]
        min_score = min(val for _, val in scored)
        best = [b for b, val in scored if val == min_score]
        if len(best) == 1:
            return best[0]
        digest = hashlib.sha256(program_id.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "big") % len(best)
        return best[idx]

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
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
    def filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_total_tokens: Callable[[int], Awaitable[None]] | None = None,
    ) -> Response:
        url = backend.completions_url

        if payload.get("stream"):
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
                total_tokens: Optional[int] = None
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
                                if total_tokens is not None:
                                    continue
                                try:
                                    payload_obj = json.loads(data)
                                except Exception:
                                    continue
                                extracted = self.extract_total_tokens(payload_obj)
                                if extracted is not None:
                                    total_tokens = extracted
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)
                    if total_tokens is not None and on_total_tokens is not None:
                        await on_total_tokens(total_tokens)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        resp = await self.client.post(url, json=payload)
        total_tokens: Optional[int] = None
        try:
            payload_obj = resp.json()
        except Exception:
            payload_obj = None
        extracted = self.extract_total_tokens(payload_obj)
        if extracted is not None:
            total_tokens = extracted
        if total_tokens is not None and on_total_tokens is not None:
            await on_total_tokens(total_tokens)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self.filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )


# =============================================================================
# FastAPI Application
# =============================================================================

router = MultiBackendRouter(VLLM_BACKENDS)
app = FastAPI(title="Multi-Backend Prefix Cache Router")


@app.on_event("startup")
async def startup_event():
    await router.start()


@app.on_event("shutdown")
async def shutdown_event():
    await router.stop()


def get_program_id(payload: Dict[str, Any], _request: Request) -> str:
    if "job_id" in payload:
        return str(payload["job_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "job_id" in extra_body:
        return str(extra_body["job_id"])
    return "default"


@app.post("/v1/chat/completions")
async def route_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = get_program_id(payload, request)

    while True:
        backend = router.get_backend_for_program(program_id)
        backend = await router.apply_pending_transfer(program_id, backend)
        async with backend.lock:
            if program_id not in backend.programs:
                backend.programs[program_id] = ProgramState(
                    context_len=0, step_count=0
                )
            state = backend.programs[program_id]

            if not state.paused:
                state.inflight = True
                state.step_count += 1
                break

            wait_event = state.resume_event
            wait_state = state
            wait_state.waiting_on_resume = True

        try:
            await wait_event.wait()
        finally:
            wait_state.waiting_on_resume = False

    async def update_total_tokens(tokens: int) -> None:
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is not None:
                state.context_len = tokens

    try:
        return await router.proxy_request(backend, payload, on_total_tokens=update_total_tokens)
    finally:
        async with backend.lock:
            if program_id in backend.programs:
                state = backend.programs[program_id]
                state.inflight = False


@app.get("/programs")
async def list_programs():
    """List all programs across all backends."""
    result = {}
    for backend in router.backends.values():
        async with backend.lock:
            for pid, s in backend.programs.items():
                result[pid] = {
                    "backend": backend.url,
                    "context_len": s.context_len,
                    "step": s.step_count,
                    "inflight": s.inflight,
                    "paused": s.paused,
                }
    return JSONResponse(result)


@app.post("/programs/release")
async def release_program(request: Request):
    """Force-release a program from router state."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = payload.get("job_id") or payload.get("program_id")
    if not program_id:
        raise HTTPException(status_code=400, detail="Missing job_id")
    program_id = str(program_id)

    released = False
    for backend in router.backends.values():
        async with backend.lock:
            if program_id in backend.programs:
                del backend.programs[program_id]
                released = True

    if program_id in router.program_affinity:
        del router.program_affinity[program_id]
        released = True

    if released:
        logger.info(f"Released program {program_id}")
    return JSONResponse({"job_id": program_id, "released": released})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")