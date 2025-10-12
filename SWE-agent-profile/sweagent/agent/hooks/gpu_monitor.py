from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory

if TYPE_CHECKING:
    from sweagent.agent.agents import DefaultAgent

try:  # pragma: no cover - optional dependency
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
    )
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover
    _NVML_AVAILABLE = False


class GpuMetricsHook(AbstractAgentHook):
    """Monitor GPU utilisation during LLM reasoning and write per-instance JSON data."""

    def __init__(self, log_file: str = "gpu_metrics.json", sample_interval: float = 0.5):
        self.log_file_name = log_file
        self.sample_interval = sample_interval
        self._log_path: Optional[Path] = None
        self._agent: DefaultAgent | None = None
        self._records: list[dict[str, Any]] = []
        self._current_samples: list[dict[str, Any]] = []
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._step_index = 0
        self._attempt_index = 1
        self._nvml_handles: list[Any] = []
        self._gpu_names: list[str] = []
        self._nvml_initialized = False

    # ------------------------- lifecycle helpers -------------------------

    def _ensure_log_path(self, *, reset: bool = False) -> None:
        if self._agent is None:
            return
        traj_path = getattr(self._agent, "traj_path", None)
        if traj_path is None:
            return
        instance_dir = Path(traj_path).parent
        instance_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = instance_dir / self.log_file_name
        if reset:
            self._records.clear()
            self._current_samples.clear()

    def _init_nvml(self) -> None:
        if not _NVML_AVAILABLE or self._nvml_initialized:
            return
        try:
            nvmlInit()
            self._nvml_initialized = True
            count = nvmlDeviceGetCount()
            for idx in range(count):
                handle = nvmlDeviceGetHandleByIndex(idx)
                self._nvml_handles.append(handle)
                try:
                    name = nvmlDeviceGetName(handle).decode("utf-8")
                except AttributeError:
                    name = str(nvmlDeviceGetName(handle))
                self._gpu_names.append(name)
        except Exception:
            self._nvml_handles = []
            self._gpu_names = []
            self._nvml_initialized = False

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except Exception:
                pass
            finally:
                self._nvml_initialized = False
                self._nvml_handles = []
                self._gpu_names = []

    # ------------------------- agent hook overrides -------------------------

    def on_init(self, *, agent: DefaultAgent):
        self._agent = agent
        self._ensure_log_path(reset=True)
        self._init_nvml()

    def on_run_start(self) -> None:
        self._step_index = 0
        self._attempt_index = 1
        self._ensure_log_path(reset=True)

    def on_setup_attempt(self) -> None:
        self._attempt_index += 1
        self._step_index = 0
        self._ensure_log_path(reset=True)

    def on_step_start(self) -> None:
        self._step_index += 1

    def on_model_query(self, *, messages, agent: str) -> None:  # type: ignore[override]
        self._start_monitoring()

    def on_actions_generated(self, *, step: StepOutput) -> None:
        self._stop_monitoring()

    def on_step_done(self, *, step: StepOutput, info: AgentInfo) -> None:
        # safety: ensure monitor stopped even if on_actions_generated not called
        self._stop_monitoring()

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo) -> None:
        self._stop_monitoring()
        self._flush_records()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self._stop_monitoring()
        self._shutdown_nvml()

    # ------------------------- monitoring logic -------------------------

    def _start_monitoring(self) -> None:
        if not self._nvml_initialized or self._monitor_thread is not None:
            return
        self._current_samples = []
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._monitor_thread.start()

    def _stop_monitoring(self) -> None:
        if self._monitor_thread is None:
            return
        self._stop_event.set()
        self._monitor_thread.join()
        self._monitor_thread = None
        if self._current_samples:
            self._records.extend(self._current_samples)
            self._current_samples = []

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._collect_snapshot()
            if snapshot is not None:
                self._current_samples.append(snapshot)
            time.sleep(max(self.sample_interval, 0.05))

    def _collect_snapshot(self) -> Optional[dict[str, Any]]:
        if not self._nvml_handles:
            return None
        gpus: list[dict[str, Any]] = []
        timestamp = time.time()
        for idx, handle in enumerate(self._nvml_handles):
            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                try:
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                gpus.append(
                    {
                        "index": idx,
                        "name": self._gpu_names[idx] if idx < len(self._gpu_names) else f"GPU-{idx}",
                        "sm_util": getattr(util, "gpu", None),
                        "mem_util": getattr(util, "memory", None),
                        "mem_used": getattr(mem, "used", None),
                        "mem_total": getattr(mem, "total", None),
                        "temperature": temp,
                    }
                )
            except Exception:
                continue
        if not gpus:
            return None
        return {
            "timestamp": timestamp,
            "step": self._step_index,
            "attempt": self._attempt_index,
            "stage": "llm_reasoning",
            "gpus": gpus,
        }

    def _flush_records(self) -> None:
        if self._log_path is None or not self._records:
            return
        try:
            with self._log_path.open("w", encoding="utf-8") as fh:
                json.dump(self._records, fh, ensure_ascii=False, indent=2)
        except Exception:
            import logging

            logging.getLogger(__name__).warning("Failed to write GPU metrics to %s", self._log_path)
