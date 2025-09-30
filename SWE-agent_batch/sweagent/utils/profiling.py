"""Utilities for collecting per-step profiling data during agent runs."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Generator


class ProfilingManager:
    """Collects wall-clock durations and per-step metrics during agent runs."""

    def __init__(self, *, problem_id: str):
        self.problem_id = problem_id
        self._stage_totals: dict[str, float] = defaultdict(float)
        self.status: str = "unknown"
        self.metadata: dict[str, object] = {}
        self._lock = threading.Lock()
        self._output_path: Path | None = None
        self._step_records: list[dict[str, object]] = []

    def set_output_path(self, output_path: Path) -> None:
        self._output_path = output_path

    def add_metadata(self, key: str, value: object) -> None:
        self.metadata[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    def set_exit_status(self, exit_status: str | None) -> None:
        self.metadata["exit_status"] = exit_status

    def set_attempts(self, attempts: int) -> None:
        self.metadata["attempts"] = attempts

    def register_group(self, parent: str, children) -> None:  # pragma: no cover - kept for compatibility
        """Groups are no longer used but we keep the method for backward compatibility."""

    def add_to_stage(self, stage: str, duration: float) -> None:
        with self._lock:
            self._stage_totals[stage] += duration

    def record_step(self, step_record: dict[str, object]) -> None:
        with self._lock:
            self._step_records.append(copy.deepcopy(step_record))

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self._stage_totals[stage] += duration

    def dump(self) -> None:
        if self._output_path is None:
            return
        with self._lock:
            timings = copy.deepcopy(self._stage_totals)
            steps = copy.deepcopy(self._step_records)
        data = {
            "problem_id": self.problem_id,
            "status": self.status,
            "timings": timings,
            "steps": steps,
            "request_totals": self._build_request_totals(timings, steps),
            "metadata": self.metadata,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(data, indent=2))

    def _build_request_totals(
        self,
        timings: dict[str, float],
        steps: list[dict[str, object]],
    ) -> dict[str, object]:
        total_llm_reasoning = 0.0
        total_llm_decode = 0.0
        total_tool = 0.0
        total_observation = 0.0
        total_decode = 0
        for step in steps:
            total_llm_decode += float(step.get("llm_decode_time", 0.0))
            total_llm_reasoning += float(step.get("llm_reasoning_time", 0.0))
            total_tool += float(step.get("tool_execution_time", 0.0))
            total_observation += float(step.get("observation_time", 0.0))
            total_decode += int(step.get("decode_tokens", 0))

        return {
            "agent_setup_time": timings.get("agent_setup", 0.0),
            "env_prepare_time": timings.get("env_preparation", 0.0),
            "total_llm_reasoning_time": total_llm_reasoning or total_llm_decode,
            "total_llm_decode_time": total_llm_decode,
            "total_tool_execution_time": total_tool,
            "total_observation_time": total_observation,
            "total_decode_tokens": total_decode,
        }


def maybe_time_stage(profiler: ProfilingManager | None, stage: str) -> ContextManager[None]:
    if profiler is None:
        return _NULL_CONTEXT
    return profiler.time_stage(stage)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


_NULL_CONTEXT = _NullContext()


class ProfilingAggregator:
    """Aggregate per-instance profiling dumps into a single summary."""

    def __init__(self):
        self._lock = threading.Lock()
        self._instances: dict[str, dict[str, object]] = {}
        self._aggregate_timings: dict[str, float] = defaultdict(float)
        self._aggregate_request_totals: dict[str, float] = defaultdict(float)
        self._step_totals: dict[str, float] = defaultdict(float)
        self._tool_usage_counts: dict[str, int] = defaultdict(int)
        self._step_count: int = 0
        self._usage_totals: dict[str, float] = {
            "tokens_sent": 0.0,
            "tokens_received": 0.0,
            "api_calls": 0.0,
            "instance_cost": 0.0,
        }

    def add(self, profiling_dump: dict[str, object]) -> None:
        with self._lock:
            instance_id = profiling_dump.get("problem_id") or profiling_dump.get("instance_id")
            if instance_id is None:
                instance_id = str(len(self._instances))
            self._instances[instance_id] = profiling_dump

            timings = profiling_dump.get("timings", {})
            if isinstance(timings, dict):
                for stage, value in timings.items():
                    if isinstance(value, (int, float)):
                        self._aggregate_timings[stage] += float(value)

            request_totals = profiling_dump.get("request_totals", {})
            if isinstance(request_totals, dict):
                for key, value in request_totals.items():
                    if isinstance(value, (int, float)):
                        self._aggregate_request_totals[key] += float(value)

            steps = profiling_dump.get("steps", [])
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    self._step_count += 1
                    for metric in ("decode_tokens",):
                        value = step.get(metric)
                        if isinstance(value, (int, float)):
                            self._step_totals[metric] += float(value)
                    for metric in (
                        "llm_reasoning_time",
                        "llm_decode_time",
                        "tool_execution_time",
                        "observation_time",
                    ):
                        value = step.get(metric)
                        if isinstance(value, (int, float)):
                            self._step_totals[metric] += float(value)
                    tool_types = step.get("tool_types", [])
                    if isinstance(tool_types, list):
                        for tool in tool_types:
                            if isinstance(tool, str):
                                self._tool_usage_counts[tool] += 1

            metadata = profiling_dump.get("metadata", {})
            if isinstance(metadata, dict):
                for key in self._usage_totals.keys():
                    value = metadata.get(key)
                    if isinstance(value, (int, float)):
                        self._usage_totals[key] += float(value)

    def has_data(self) -> bool:
        with self._lock:
            return bool(self._instances)

    def apply_evaluation_results(self, resolved_ids: set[str], failed_ids: set[str] | None = None) -> None:
        failed_ids = failed_ids or set()
        with self._lock:
            for instance_id, data in self._instances.items():
                if instance_id in resolved_ids:
                    data["status"] = "success"
                elif instance_id in failed_ids:
                    data["status"] = "failure"
                else:
                    data["status"] = "failure"

    def summary(self) -> dict[str, object] | None:
        with self._lock:
            if not self._instances:
                return None

            instances_list = list(self._instances.values())
            instance_count = len(instances_list)

            timings_total = copy.deepcopy(self._aggregate_timings)
            timings_average = {
                stage: (value / instance_count if instance_count else 0.0)
                for stage, value in timings_total.items()
            }

            request_totals_total = copy.deepcopy(self._aggregate_request_totals)
            request_totals_average = {
                key: (value / instance_count if instance_count else 0.0)
                for key, value in request_totals_total.items()
            }

            step_metrics_average: dict[str, float] = {}
            if self._step_count:
                for key, value in self._step_totals.items():
                    step_metrics_average[key] = value / self._step_count
            else:
                step_metrics_average = {key: 0.0 for key in self._step_totals.keys()}

            usage_totals = {
                key: (int(value) if float(value).is_integer() else value)
                for key, value in self._usage_totals.items()
            }
            usage_average = {
                key: (usage_totals[key] / instance_count if instance_count else 0.0)
                for key in usage_totals.keys()
            }
            for key, value in usage_average.items():
                if float(value).is_integer():
                    usage_average[key] = int(value)

            success_count = sum(1 for data in instances_list if data.get("status") == "success")
            success_rate = {
                "successful_instances": success_count,
                "failed_instances": instance_count - success_count,
                "percentage": (success_count / instance_count) if instance_count else 0.0,
            }

            step_metrics_total: dict[str, object] = {}
            for key, value in self._step_totals.items():
                if key.endswith("_tokens"):
                    step_metrics_total[key] = int(round(value))
                else:
                    step_metrics_total[key] = value

            request_totals_total_cast: dict[str, object] = {}
            for key, value in request_totals_total.items():
                if key.endswith("_tokens"):
                    request_totals_total_cast[key] = int(round(value))
                else:
                    request_totals_total_cast[key] = value

            request_totals_average_cast: dict[str, object] = {}
            for key, value in request_totals_average.items():
                request_totals_average_cast[key] = value

            aggregate = {
                "timings_total": timings_total,
                "timings_average": timings_average,
                "request_totals_total": request_totals_total_cast,
                "request_totals_average": request_totals_average_cast,
                "step_metrics_total": step_metrics_total,
                "step_metrics_average": step_metrics_average,
                "tool_usage_counts": dict(self._tool_usage_counts),
                "usage_totals": usage_totals,
                "usage_average": usage_average,
                "step_count": self._step_count,
            }

            return copy.deepcopy(
                {
                    "instances_count": instance_count,
                    "aggregate": aggregate,
                    "success_rate": success_rate,
                    "instances": instances_list,
                }
            )
