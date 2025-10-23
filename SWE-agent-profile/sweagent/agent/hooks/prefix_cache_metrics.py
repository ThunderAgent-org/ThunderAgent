"""Hook to track LLM reasoning time per step and pull vLLM prefix cache metrics."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests

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
    {"step": 0, "timestamp": 1234567890.123, "prefix_cache_hits_before": 100, "prefix_cache_hits_after": 120,
     "prefix_cache_queries_before": 200, "prefix_cache_queries_after": 240, ...}
    
    The metrics URL can be set via:
    1. Explicit constructor parameter
    2. VLLM_METRICS_URL environment variable
    3. Default: http://localhost:8000/metrics
    """

    def __init__(self, metrics_url: str | None = None, log_file: str = "prefix_cache_metrics.jsonl"):
        self.metrics_url = metrics_url
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
        self._resolved_metrics_url: str | None = None
        self._metrics_before_snapshot: dict[str, float | None] | None = None
        self._metrics_failure_logged = False

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
        self._metrics_before_snapshot = None
        model = getattr(self._agent, "model", None)
        if model is not None:
            setattr(model, "_last_pd_metrics", None)
            setattr(model, "_last_pd_request_id", None)

    def on_step_done(self, *, step: StepOutput, info: AgentInfo):
        """Finalize step bookkeeping."""
        self._reasoning_start = None
        self._prefill_start = None
        self._prefill_start_wall = None
        self._first_token_time = None
        self._first_token_wall_time = None
        self._llm_duration_accum = 0.0
        self._metrics_before_snapshot = None
        model = getattr(self._agent, "model", None)
        if model is not None:
            setattr(model, "_last_pd_metrics", None)
            setattr(model, "_last_pd_request_id", None)

    def on_model_query(self, *, messages, agent: str) -> None:  # type: ignore[override]
        self._reasoning_start = time.perf_counter()
        self._prefill_start = self._reasoning_start
        self._prefill_start_wall = time.time()
        self._first_token_time = None
        self._first_token_wall_time = None
        self._metrics_before_snapshot = self._fetch_prefix_cache_snapshot()

    def on_actions_generated(self, *, step: StepOutput) -> None:
        if self._reasoning_start is not None:
            self._llm_duration_accum += time.perf_counter() - self._reasoning_start
            self._reasoning_start = None
        after_snapshot = self._fetch_prefix_cache_snapshot()
        before_snapshot = self._metrics_before_snapshot
        self._metrics_before_snapshot = None
        self._write_record(before_snapshot, after_snapshot)

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

    def _write_record(
        self,
        snapshot_before: dict[str, float | None] | None,
        snapshot_after: dict[str, float | None] | None,
    ) -> None:
        if self.log_path is None:
            return
        model = getattr(self._agent, "model", None)
        decode_request_wall = getattr(model, "_last_decode_request_wall_start", None) if model is not None else None
        pd_metrics = getattr(model, "_last_pd_metrics", None) if model is not None else None
        if isinstance(pd_metrics, dict):
            prefill_start_wall = pd_metrics.get("prefill_start")
            decode_start_wall = pd_metrics.get("decode_start")
            decode_end_wall = pd_metrics.get("decode_end")
            prefill_end_wall = decode_start_wall
            decode_request_wall = pd_metrics.get("decode_request_start", decode_request_wall)
        else:
            prefill_start_wall = self._prefill_start_wall
            decode_start_wall = getattr(model, "_last_first_chunk_wall", None) if model is not None else None
            decode_end_wall = getattr(model, "_last_chunk_wall_time", None) if model is not None else None
            prefill_end_wall = decode_start_wall

        def _non_negative_delta(start: float | None, end: float | None) -> float:
            if start is None or end is None:
                return 0.0
            return max(0.0, end - start)

        prefill_time = _non_negative_delta(prefill_start_wall, prefill_end_wall)
        decode_wait_time = _non_negative_delta(decode_request_wall, decode_start_wall)
        decode_time = _non_negative_delta(decode_start_wall, decode_end_wall)

        input_tokens = getattr(model, "_last_input_tokens", None)
        output_tokens = getattr(model, "_last_output_tokens", None)
        current_time = time.time()
        record = {
            "step": self.current_step,
            "timestamp": current_time,
            "prefill_start_timestamp": prefill_start_wall,
            "prefill_end_timestamp": prefill_end_wall,
            "decode_request_start_timestamp": decode_request_wall,
            "decode_start_timestamp": decode_start_wall,
            "decode_end_timestamp": decode_end_wall,
            "first_chunk_timestamp": decode_start_wall,
            "llm_reasoning_time": self._llm_duration_accum,
            "prefill_time": prefill_time,
            "decode_wait_time": decode_wait_time,
            "decode_time": decode_time,
        }
        pd_request_id = getattr(model, "_last_pd_request_id", None) if model is not None else None
        if pd_request_id is not None:
            record["pd_request_id"] = pd_request_id
        if input_tokens is not None:
            record["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            record["output_tokens"] = int(output_tokens)
        if isinstance(pd_metrics, dict) and "prefill_http_end" in pd_metrics:
            record["prefill_http_end_timestamp"] = pd_metrics.get("prefill_http_end")

        if snapshot_before is not None:
            record["metrics_before_timestamp"] = snapshot_before.get("timestamp")
            if snapshot_before.get("hits") is not None:
                record["prefix_cache_hits_before"] = snapshot_before["hits"]
            if snapshot_before.get("queries") is not None:
                record["prefix_cache_queries_before"] = snapshot_before["queries"]

        if snapshot_after is not None:
            record["metrics_after_timestamp"] = snapshot_after.get("timestamp")
            if snapshot_after.get("hits") is not None:
                record["prefix_cache_hits_after"] = snapshot_after["hits"]
            if snapshot_after.get("queries") is not None:
                record["prefix_cache_queries_after"] = snapshot_after["queries"]

        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logging.getLogger(__name__).warning(f"Failed to write prefix cache metrics to {self.log_path}: {exc}")

    def _on_first_token(self, perf_time: float, wall_time: float) -> None:
        self._first_token_time = perf_time
        self._first_token_wall_time = wall_time

    def _get_metrics_url(self) -> str:
        if self._resolved_metrics_url is None:
            env_url = os.getenv("VLLM_METRICS_URL")
            url = self.metrics_url or env_url or "http://localhost:8000/metrics"
            self._resolved_metrics_url = url
        return self._resolved_metrics_url

    def _fetch_prefix_cache_snapshot(self) -> dict[str, float | None] | None:
        url = self._get_metrics_url()
        if not url:
            return None
        try:
            response = requests.get(url, timeout=1.5)
            response.raise_for_status()
        except requests.RequestException as exc:
            self._log_metrics_warning(f"Failed to query vLLM metrics at {url}: {exc}")
            return None

        hits, queries = self._parse_prefix_cache_metrics(response.text)
        if hits is None and queries is None:
            self._log_metrics_warning(f"vLLM metrics at {url} did not include prefix cache counters.")
            return None

        self._metrics_failure_logged = False
        return {
            "timestamp": time.time(),
            "hits": hits,
            "queries": queries,
        }

    def _parse_prefix_cache_metrics(self, payload: str) -> tuple[float | None, float | None]:
        metrics: dict[str, float] = {}
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                metric_and_labels, value_str = line.rsplit(None, 1)
            except ValueError:
                continue
            metric_name = metric_and_labels.split("{", 1)[0]
            if metric_name.endswith("_created"):
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            metrics[metric_name] = metrics.get(metric_name, 0.0) + value

        hits = self._select_metric_sum(
            metrics,
            [
                ("vllm:prefix_cache_hits_total",),
                ("vllm_prefix_cache_hits_total",),
                ("vllm:prefix_cache_hits",),
                ("vllm_prefix_cache_hits",),
                ("vllm:gpu_prefix_cache_hits_total",),
                ("vllm_gpu_prefix_cache_hits_total",),
                ("vllm:gpu_prefix_cache_hits",),
                ("vllm_gpu_prefix_cache_hits",),
            ],
        )
        queries = self._select_metric_sum(
            metrics,
            [
                ("vllm:prefix_cache_queries_total",),
                ("vllm_prefix_cache_queries_total",),
                ("vllm:prefix_cache_queries",),
                ("vllm_prefix_cache_queries",),
                ("vllm:gpu_prefix_cache_queries_total",),
                ("vllm_gpu_prefix_cache_queries_total",),
                ("vllm:gpu_prefix_cache_queries",),
                ("vllm_gpu_prefix_cache_queries",),
            ],
        )
        return hits, queries

    @staticmethod
    def _select_metric_sum(
        metrics: dict[str, float],
        candidates: list[tuple[str, ...]],
    ) -> float | None:
        for names in candidates:
            values = [metrics[name] for name in names if name in metrics]
            if values:
                return sum(values)
        return None

    def _log_metrics_warning(self, message: str) -> None:
        if not self._metrics_failure_logged:
            logging.getLogger(__name__).warning(message)
            self._metrics_failure_logged = True
