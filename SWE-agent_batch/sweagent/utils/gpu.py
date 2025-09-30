"""Helpers for sampling GPU utilisation during agent runs."""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_GPU_QUERY = [
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
]


@dataclass
class _MetricAggregate:
    count: int = 0
    util_gpu_sum: float = 0.0
    util_mem_sum: float = 0.0
    mem_used_sum: float = 0.0
    util_gpu_max: float = 0.0
    util_mem_max: float = 0.0
    mem_used_max: float = 0.0

    def update(self, util_gpu: float, util_mem: float, mem_used: float) -> None:
        self.count += 1
        self.util_gpu_sum += util_gpu
        self.util_mem_sum += util_mem
        self.mem_used_sum += mem_used
        self.util_gpu_max = max(self.util_gpu_max, util_gpu)
        self.util_mem_max = max(self.util_mem_max, util_mem)
        self.mem_used_max = max(self.mem_used_max, mem_used)

    def as_dict(self) -> Dict[str, float]:
        if self.count:
            util_gpu_avg = self.util_gpu_sum / self.count
            util_mem_avg = self.util_mem_sum / self.count
            mem_used_avg = self.mem_used_sum / self.count
        else:
            util_gpu_avg = util_mem_avg = mem_used_avg = 0.0
        return {
            "average_gpu_util": util_gpu_avg,
            "average_mem_util": util_mem_avg,
            "average_mem_used_mb": mem_used_avg,
            "peak_gpu_util": self.util_gpu_max,
            "peak_mem_util": self.util_mem_max,
            "peak_mem_used_mb": self.mem_used_max,
            "samples": self.count,
        }


@dataclass
class GPUMonitor:
    """Background sampler for NVIDIA GPU utilisation via ``nvidia-smi``."""

    interval: float = 1.0
    timeout: float = 2.0
    _thread: Optional[threading.Thread] = field(init=False, default=None)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _data: Dict[str, _MetricAggregate] = field(init=False, default_factory=lambda: defaultdict(_MetricAggregate))
    _names: Dict[str, str] = field(init=False, default_factory=dict)
    _memory_totals: Dict[str, float] = field(init=False, default_factory=dict)
    _available: bool = field(init=False, default=False)
    _samples: Dict[str, List[Tuple[float, float, float]]] = field(init=False, default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def start(self) -> bool:
        if self._thread is not None:
            return self._available
        if shutil.which("nvidia-smi") is None:
            return False
        self._available = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Dict[str, Dict[str, float]]:
        if not self._available:
            return {}
        assert self._thread is not None
        self._stop.set()
        self._thread.join()
        self._thread = None
        return self.summary()

    def snapshot(self) -> Dict[str, int]:
        if not self._available:
            return {}
        with self._lock:
            return {uuid: len(samples) for uuid, samples in self._samples.items()}

    def segment_summary(self, start: Optional[Dict[str, int]], end: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, float]]:
        if not self._available:
            return {}
        with self._lock:
            effective_end = end or {uuid: len(samples) for uuid, samples in self._samples.items()}
            result: Dict[str, Dict[str, float]] = {}
            for uuid, samples in self._samples.items():
                start_idx = 0
                if start is not None:
                    start_idx = start.get(uuid, 0)
                end_idx = effective_end.get(uuid, len(samples))
                segment = samples[start_idx:end_idx]
                if not segment:
                    continue
                util_gpu_vals = [s[0] for s in segment]
                util_mem_vals = [s[1] for s in segment]
                mem_used_vals = [s[2] for s in segment]
                entry: Dict[str, float] = {
                    "average_gpu_util": sum(util_gpu_vals) / len(segment),
                    "average_mem_util": sum(util_mem_vals) / len(segment),
                    "average_mem_used_mb": sum(mem_used_vals) / len(segment),
                    "peak_gpu_util": max(util_gpu_vals),
                    "peak_mem_util": max(util_mem_vals),
                    "peak_mem_used_mb": max(mem_used_vals),
                    "samples": len(segment),
                }
                if uuid in self._names:
                    entry["name"] = self._names[uuid]
                if uuid in self._memory_totals:
                    entry["memory_total_mb"] = self._memory_totals[uuid]
                result[uuid] = entry
            return result

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for uuid, aggregate in self._data.items():
                entry = aggregate.as_dict()
                if uuid in self._names:
                    entry["name"] = self._names[uuid]
                if uuid in self._memory_totals:
                    entry["memory_total_mb"] = self._memory_totals[uuid]
                result[uuid] = entry
        return result

    @property
    def available(self) -> bool:
        return self._available

    # Internal

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def _sample_once(self) -> None:
        try:
            cmd = [
                "nvidia-smi",
                f"--query-gpu={','.join(_GPU_QUERY)}",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except Exception:
            return
        if completed.returncode != 0 or not completed.stdout:
            return
        for line in completed.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(_GPU_QUERY):
                continue
            uuid, name, util_gpu, util_mem, mem_used, mem_total = parts
            try:
                util_gpu_f = float(util_gpu)
                util_mem_f = float(util_mem)
                mem_used_f = float(mem_used)
            except ValueError:
                continue
            with self._lock:
                aggregate = self._data[uuid]
                aggregate.update(util_gpu_f, util_mem_f, mem_used_f)
                self._samples[uuid].append((util_gpu_f, util_mem_f, mem_used_f))
                if uuid not in self._names:
                    self._names[uuid] = name
                if uuid not in self._memory_totals:
                    try:
                        self._memory_totals[uuid] = float(mem_total)
                    except ValueError:
                        pass