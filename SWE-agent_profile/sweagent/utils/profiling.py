"""Utilities for collecting stage-level profiling data during agent runs."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Generator, Iterable


def _is_group(node: object) -> bool:
    return isinstance(node, dict) and "total" in node and "breakdown" in node


def _merge_stage_maps(dest: dict[str, object], src: dict[str, object]) -> None:
    for name, value in src.items():
        if _is_group(value):
            node = dest.setdefault(name, {"total": 0.0, "breakdown": {}})
            assert isinstance(node, dict)
            node["total"] = float(node.get("total", 0.0)) + float(value.get("total", 0.0))
            _merge_stage_maps(node.setdefault("breakdown", {}), value.get("breakdown", {}))
        else:
            dest[name] = float(dest.get(name, 0.0)) + float(value)


def _stage_total(value: object) -> float:
    if _is_group(value):
        return float(value.get("total", 0.0))
    return float(value)


def _sum_top_levels(stage_totals: dict[str, object]) -> float:
    return sum(_stage_total(value) for value in stage_totals.values())


def _compute_percentages(stage_totals: dict[str, object], parent_total: float) -> dict[str, object]:
    percentages: dict[str, object] = {}
    for name, value in stage_totals.items():
        total = _stage_total(value)
        percentage = total / parent_total if parent_total else 0.0
        if _is_group(value):
            breakdown = value.get("breakdown", {})
            child_percentages = _compute_percentages(
                breakdown,
                total if total else 1.0,
            )
            percentages[name] = {"percentage": percentage, "breakdown": child_percentages}
        else:
            percentages[name] = percentage
    return percentages


class ProfilingManager:
    """Collects wall-clock durations for named stages and writes them to disk."""

    def __init__(self, *, problem_id: str):
        self.problem_id = problem_id
        self._stage_totals: dict[str, float] = defaultdict(float)
        self.status: str = "unknown"
        self.metadata: dict[str, object] = {}
        self._lock = threading.Lock()
        self._output_path: Path | None = None
        self._groups: dict[str, tuple[str, ...]] = {}
        self._child_to_parent: dict[str, str] = {}
        self._stage_order: list[str] = []

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

    def register_group(self, parent: str, children: Iterable[str]) -> None:
        children = tuple(children)
        self._groups[parent] = children
        if parent not in self._stage_order:
            self._stage_order.append(parent)
        for child in children:
            if child in self._child_to_parent:
                raise ValueError(f"Stage '{child}' already has a parent '{self._child_to_parent[child]}'")
            self._child_to_parent[child] = parent

    def add_to_stage(self, stage: str, duration: float) -> None:
        with self._lock:
            if stage not in self._stage_order:
                self._stage_order.append(stage)
            self._stage_totals[stage] += duration

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                if stage not in self._stage_order:
                    self._stage_order.append(stage)
                self._stage_totals[stage] += duration

    def dump(self) -> None:
        if self._output_path is None:
            return
        stage_totals = self._build_stage_output()
        data = {
            "problem_id": self.problem_id,
            "status": self.status,
            "stage_totals": stage_totals,
            "metadata": self.metadata,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(data, indent=2))

    # Helpers

    def _build_stage_output(self) -> dict[str, object]:
        roots = []
        for stage in self._stage_order:
            if stage not in self._child_to_parent:
                roots.append(stage)

        output: dict[str, object] = {}
        for stage in roots:
            structure, _ = self._stage_structure(stage)
            if structure is None:
                continue
            output[stage] = structure
        return output

    def _stage_structure(self, stage: str) -> tuple[object | None, float]:
        is_group = stage in self._groups
        total = self._stage_totals.get(stage, 0.0)
        if not is_group:
            return (total if total else None, total)

        breakdown: dict[str, object] = {}
        child_sum = 0.0
        for child in self._groups[stage]:
            structure, child_total = self._stage_structure(child)
            if structure is None and child_total == 0.0:
                continue
            breakdown[child] = structure
            child_sum += child_total

        if total == 0.0:
            total = child_sum

        if total > 0.0:
            remaining = total - child_sum
            if remaining > 1e-9:
                breakdown["other_time"] = remaining

        if not breakdown and total == 0.0:
            return (None, 0.0)

        structure: dict[str, object] = {"total": total, "breakdown": breakdown}
        return (structure, total)


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
        self._aggregate: dict[str, object] = {}
        self._usage_totals: dict[str, float] = {
            "tokens_sent": 0.0,
            "tokens_received": 0.0,
            "api_calls": 0.0,
            "instance_cost": 0.0,
        }
        self._success_count: int = 0

    def add(self, profiling_dump: dict[str, object]) -> None:
        with self._lock:
            instance_id = profiling_dump.get("problem_id") or profiling_dump.get("instance_id")
            if instance_id is None:
                instance_id = str(len(self._instances))
            self._instances[instance_id] = profiling_dump
            stage_totals = profiling_dump.get("stage_totals", {})
            if isinstance(stage_totals, dict):
                _merge_stage_maps(self._aggregate, stage_totals)
            metadata = profiling_dump.get("metadata", {})
            if isinstance(metadata, dict):
                for key in self._usage_totals.keys():
                    value = metadata.get(key)
                    if isinstance(value, (int, float)):
                        self._usage_totals[key] += float(value)
            if profiling_dump.get("status") == "success":
                self._success_count += 1

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
            total_time = _sum_top_levels(self._aggregate)
            percentages = _compute_percentages(self._aggregate, total_time if total_time else 1.0)
            usage_totals = {
                key: (int(value) if float(value).is_integer() else value)
                for key, value in self._usage_totals.items()
            }
            instances_list = list(self._instances.values())
            count = len(instances_list)
            usage_average = {
                key: (
                    (usage_totals[key] / count)
                    if count
                    else 0.0
                )
                for key in usage_totals.keys()
            }
            for key, value in usage_average.items():
                if float(value).is_integer():
                    usage_average[key] = int(value)
            success_count = sum(1 for data in instances_list if data.get("status") == "success")
            success_rate = {
                "successful_instances": success_count,
                "failed_instances": count - success_count,
                "percentage": (success_count / count) if count else 0.0,
            }
            return copy.deepcopy({
                "instances_count": count,
                "aggregate": {
                    "total_time": total_time,
                    "stage_totals": self._aggregate,
                    "percentages": percentages,
                    "usage_average": usage_average,
                },
                "success_rate": success_rate,
                "instances": instances_list,
            })
