"""Hook to track LLM reasoning time per step (without querying vLLM metrics)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput

if TYPE_CHECKING:
    from sweagent.agent.agents import DefaultAgent


class PrefixCacheMetricsHook(AbstractAgentHook):
    """
    Queries vLLM /metrics endpoint around each model call and logs prefix cache hit rate.
    
    For every agent step, the hook captures a baseline snapshot before the model
    query and another afterwards to compute per-step deltas for prefix cache
    usage and prefill/decode timings. Logs are saved to <instance_dir>/prefix_cache_metrics.jsonl with format:
    {"step": 0, "timestamp": 1234567890.123, "prefix_cache_hits": 100, "prefix_cache_queries": 200, "hit_rate": 0.5}
    
    The metrics URL can be set via:
    1. Explicit constructor parameter
    2. VLLM_METRICS_URL environment variable
    3. Default: http://localhost:8000/metrics
    """

    def __init__(self, metrics_url: str | None = None, log_file: str = "prefix_cache_metrics.jsonl"):
        self.metrics_url = metrics_url  # kept for backwards compatibility (unused)
        self.log_file_name = log_file
        self.log_path: Path | None = None
        self.current_step = 0
        self._agent: DefaultAgent | None = None
        self._step_counter = 0
        self._reasoning_start: float | None = None
        self._prefill_start: float | None = None
        self._prefill_start_wall: float | None = None
        self._first_token_time: float | None = None
        self._first_token_wall_time: float | None = None
        self._llm_duration_accum: float = 0.0

    def on_init(self, *, agent: DefaultAgent):
        self._agent = agent
        model = getattr(agent, "model", None)
        if model is not None and hasattr(model, "register_first_token_callback"):
            model.register_first_token_callback(self._on_first_token)

    def on_run_start(self):
        self._compute_log_path(reset_step=True)

    def on_setup_attempt(self):
        # In retry mode, a new attempt uses a new output directory. Recompute to be safe.
        self._compute_log_path(reset_step=True)

    def on_step_start(self):
        if self.log_path is None:
            self._compute_log_path()
        self._step_counter += 1
        self.current_step = self._step_counter
        self._reasoning_start = None
        self._prefill_start = None
        self._prefill_start_wall = None
        self._first_token_time = None
        self._first_token_wall_time = None
        self._llm_duration_accum = 0.0

    def on_step_done(self, *, step: StepOutput, info: AgentInfo):
        """Finalize step bookkeeping."""
        self._reasoning_start = None
        self._prefill_start = None
        self._prefill_start_wall = None
        self._first_token_time = None
        self._first_token_wall_time = None
        self._llm_duration_accum = 0.0

    def on_model_query(self, *, messages, agent: str) -> None:  # type: ignore[override]
        self._reasoning_start = time.perf_counter()
        self._prefill_start = self._reasoning_start
        self._prefill_start_wall = time.time()
        self._first_token_time = None
        self._first_token_wall_time = None

    def on_actions_generated(self, *, step: StepOutput) -> None:
        if self._reasoning_start is not None:
            self._llm_duration_accum += time.perf_counter() - self._reasoning_start
            self._reasoning_start = None
        self._write_record()

    def _compute_log_path(self, *, reset_step: bool = False):
        agent = self._agent
        if agent is None:
            return
        traj_path = getattr(agent, "traj_path", None)
        if traj_path is None:
            return
        instance_dir = Path(traj_path).parent
        instance_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = instance_dir / self.log_file_name
        if reset_step:
            self.current_step = 0
            self._step_counter = 0
            self._reasoning_start = None
            self._prefill_start = None
            self._llm_duration_accum = 0.0

    def _write_record(self) -> None:
        if self.log_path is None:
            return
        model = getattr(self._agent, "model", None)
        prefill_time = 0.0
        first_chunk_wall = getattr(model, "_last_first_chunk_wall", None) if model is not None else None
        if self._prefill_start_wall is not None and first_chunk_wall is not None:
            prefill_time = max(0.0, first_chunk_wall - self._prefill_start_wall)
        last_chunk_wall = getattr(model, "_last_chunk_wall_time", None) if model is not None else None
        if first_chunk_wall is not None and last_chunk_wall is not None:
            decode_time = max(0.0, last_chunk_wall - first_chunk_wall)
        else:
            decode_time = 0.0
        input_tokens = getattr(model, "_last_input_tokens", None)
        output_tokens = getattr(model, "_last_output_tokens", None)
        record = {
            "step": self.current_step,
            "first_chunk_timestamp": first_chunk_wall,
            "llm_reasoning_time": self._llm_duration_accum,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
        }
        if input_tokens is not None:
            record["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            record["output_tokens"] = int(output_tokens)
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to write prefix cache metrics to {self.log_path}: {exc}")

    def _on_first_token(self, perf_time: float, wall_time: float) -> None:
        self._first_token_time = perf_time
        self._first_token_wall_time = wall_time
