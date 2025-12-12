import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer, PreTrainedTokenizerBase

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8100")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", f"{VLLM_API_BASE}/metrics")
PAUSE_AT_USAGE = 0.95
RESUME_AT_USAGE = 0.9
KV_CACHE_TOKEN_BUDGET = 1005312 
OUTPUT_TOKEN_ESTIMATE = 500
LOW_USAGE_RELEASE = 0.5  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, PreTrainedTokenizerBase] = {}

@dataclass
class ProgramState:
    context_len: int
    step_count: int = 0
    inflight: int = 0
    paused: bool = False
    pause_requested: bool = False
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def est_tokens(self) -> int:
        return self.context_len + OUTPUT_TOKEN_ESTIMATE

    def __post_init__(self):
        if not self.paused:
            self.resume_event.set()

class PrefixCacheRouter:
    def __init__(self) -> None:
        self.programs: Dict[str, ProgramState] = {}
        self.lock = asyncio.Lock()
        # One client for forwarding (no connection limits), one lightweight client for metrics.
        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.metrics_client = httpx.AsyncClient(timeout=5.0)
        self.current_usage: float = 0.0
        self.monitor_task: Optional[asyncio.Task] = None
        self.last_resume_time: float = 0.0
        self.last_pause_time: float = 0.0
        self.high_usage_hits: int = 0  # consecutive high-usage samples

    async def start(self):
        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
        await self.metrics_client.aclose()

    async def _monitor_loop(self):
        while True:
            try:
                usage = await self._fetch_usage()
                async with self.lock:
                    self.current_usage = usage
                    self._update_pause_state(usage)
            except Exception as exc:  # noqa: BLE001
                # Keep previous usage; log for observability.
                logger.exception(
                    "Failed to refresh usage metrics from %s: %r",
                    VLLM_METRICS_URL,
                    exc,
                )
            await asyncio.sleep(0.5)

    async def _fetch_usage(self) -> float:
        resp = await self.metrics_client.get(VLLM_METRICS_URL, timeout=3.0)
        resp.raise_for_status()
        metric_names = (
            "vllm:kv_cache_usage_perc",
        )
        for family in text_string_to_metric_families(resp.text):
            if family.name in metric_names:
                for sample in family.samples:
                    if sample.name == family.name:
                        val = float(sample.value)
                        return val / 100.0 if val > 1.0 else val
        return 0.0

    def estimate_tokens(self, text: str, model_name: str) -> int:
        try:
            if model_name not in _TOKENIZER_CACHE:
                _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
            tok = _TOKENIZER_CACHE[model_name]
            return len(tok.encode(text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback token estimate for model %s: %s",
                           model_name, exc)
            return len(text) // 4

    def _pick_to_resume(self, usage: float) -> Tuple[List[str], int]:
        # Rate limit: Don't resume too much at once. 
        # Cap the resume budget based on current usage.
        if usage < 0.5:
            step_pct = 0.20
        elif usage < 0.6:
            step_pct = 0.10
        elif usage < 0.75:
            step_pct = 0.05
        elif usage < 0.8:
            step_pct = 0.03
        elif usage < 0.85:
            step_pct = 0.02
        else:
            step_pct = 0.01

        max_resume_tokens = KV_CACHE_TOKEN_BUDGET * step_pct
        tokens_needed = max_resume_tokens

        if tokens_needed <= 0:
            # If usage is very low (< 0.5) but tokens_needed is <= 0 (which shouldn't happen with the new logic, 
            # but just in case of floating point weirdness or extreme edge cases), 
            # force a small positive budget to ensure we resume at least something.
            if usage < LOW_USAGE_RELEASE:
                tokens_needed = 1.0 # Ensure we enter the loop
            else:
                return [], 0

        paused = [
            (pid, state)
            for pid, state in self.programs.items()
            if state.paused
        ]
        paused.sort(key=lambda item: (-item[1].step_count, item[1].est_tokens))
        
        added = 0
        resumed_pids = []
        for pid, state in paused:
            if added >= tokens_needed:
                break
            added += state.est_tokens
            resumed_pids.append(pid)
            
        return resumed_pids, int(tokens_needed)

    def _update_pause_state(self, usage: float):
        now = time.time()
        if usage >= PAUSE_AT_USAGE:
            self.high_usage_hits += 1
            if self.high_usage_hits < 2:
                return
            if now - self.last_pause_time < 2.5:
                return
            self.last_pause_time = now
            self._handle_high_usage(usage)
        elif usage < RESUME_AT_USAGE:
            self.high_usage_hits = 0
            # Only check resume every 10 seconds to prevent flapping
            if now - self.last_resume_time < 10.0:
                return
            self.last_resume_time = now

            # Resume paused programs, prioritizing higher step_count.
            resumed_pids, budget = self._pick_to_resume(usage)
            
            for pid in resumed_pids:
                state = self.programs[pid]
                state.paused = False
                state.pause_requested = False
                state.resume_event.set()
            
            if resumed_pids:
                resume_details = [
                    f"{pid}(ctx={self.programs[pid].context_len}, step={self.programs[pid].step_count})"
                    for pid in resumed_pids
                ]
                logger.info(
                    "Resumed programs (usage=%.2f, budget=%d): %s", 
                    usage, budget, resume_details
                )
        else:
            self.high_usage_hits = 0
            # In-between zone: no action.
            return

    def _handle_high_usage(self, current_usage: float):
        """When usage >= 95%, pause programs to reduce usage by ~1%."""
        active = [
            (pid, state)
            for pid, state in self.programs.items()
            if not state.paused and not state.pause_requested
        ]
        if not active:
            return

        # Calculate tokens to pause: 1% of total budget
        tokens_to_pause = KV_CACHE_TOKEN_BUDGET * 0.01
        
        # Sort: keep highest step count, smallest tokens.
        # So the ones to pause (victims) are at the end (lowest priority).
        active.sort(key=lambda item: (-item[1].step_count, item[1].est_tokens))

        paused_count = 0
        paused_tokens = 0
        newly_paused = []

        # Iterate from the end (lowest priority)
        for i in range(len(active) - 1, -1, -1):
            if paused_tokens >= tokens_to_pause:
                break
            
            pid, state = active[i]
            
            if state.inflight > 0:
                state.pause_requested = True
            else:
                state.paused = True
                state.resume_event.clear()
            
            paused_tokens += state.est_tokens
            newly_paused.append(pid)
            paused_count += 1
        
        if newly_paused:
            logger.info(
                "Paused %d programs (target ~%d tokens, actual %d): %s", 
                paused_count, int(tokens_to_pause), paused_tokens, newly_paused
            )

    @staticmethod
    def _filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    async def proxy_request(self, payload: Dict[str, Any]) -> Response:
        url = f"{VLLM_API_BASE}/v1/chat/completions"
        if payload.get("stream"):
            # Use a stream context and explicitly close it after iteration.
            resp_cm = self.client.stream("POST", url, json=payload)
            resp = await resp_cm.__aenter__()
            headers = self._filtered_headers(resp.headers)
            status = resp.status_code
            media_type = resp.headers.get("content-type")

            async def iterator():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        resp = await self.client.post(url, json=payload)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self._filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )

router = PrefixCacheRouter()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await router.start()

@app.on_event("shutdown")
async def shutdown_event():
    await router.stop()

def _get_program_id(payload: Dict[str, Any], request: Request) -> str:
    if "x-program-id" in request.headers:
        return request.headers["x-program-id"]
    if "router_program_id" in payload:
        return str(payload["router_program_id"])
    if "job_id" in payload:
        return str(payload["job_id"])
    return "default"

@app.post("/v1/chat/completions")
async def route_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = _get_program_id(payload, request)
    model_name = payload.get("model", "default")
    
    prompt_text = ""
    if "messages" in payload:
        for m in payload["messages"]:
            prompt_text += str(m.get("content", ""))
    elif "prompt" in payload:
        p = payload["prompt"]
        prompt_text = p if isinstance(p, str) else "".join(str(x) for x in p)
    
    input_tokens = router.estimate_tokens(prompt_text, model_name)
    
    extra_body = payload.get("extra_body", {})
    is_last_step = extra_body.get("is_last_step", False) if extra_body else False

    first_pass = True
    while True:
        async with router.lock:
            if program_id not in router.programs:
                router.programs[program_id] = ProgramState(context_len=input_tokens, step_count=0)
            state = router.programs[program_id]
            state.context_len = input_tokens
            
            if first_pass:
                # Update step count: prefer explicit step in extra_body, else increment.
                step_val = None
                if isinstance(extra_body, dict):
                    raw_step = extra_body.get("step")
                    if isinstance(raw_step, int) and raw_step >= 0:
                        step_val = raw_step
                if step_val is not None:
                    state.step_count = max(state.step_count, step_val)
                else:
                    state.step_count += 1
                first_pass = False

            usage = router.current_usage

            if not state.paused:
                state.inflight += 1
                break
            
            wait_event = state.resume_event
        
        # Wait for resume
        await wait_event.wait()

    try:
        return await router.proxy_request(payload)
    finally:
        async with router.lock:
            if program_id in router.programs:
                state = router.programs[program_id]
                state.inflight = max(0, state.inflight - 1)
                
                if is_last_step:
                    del router.programs[program_id]
                elif state.pause_requested and state.inflight == 0:
                    state.paused = True
                    state.resume_event.clear()

@app.get("/programs")
async def list_programs():
    async with router.lock:
        return JSONResponse({
            pid: {
                "context_len": s.context_len,
                "step": s.step_count,
                "inflight": s.inflight,
                "paused": s.paused,
                "pause_requested": s.pause_requested,
            }
            for pid, s in router.programs.items()
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="error")

