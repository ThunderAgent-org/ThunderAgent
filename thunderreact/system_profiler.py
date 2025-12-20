#!/usr/bin/env python3
"""
System Resource Profiler - 系统资源监控器

监控 GPU、CPU、内存和磁盘使用情况，每 0.5 秒采样一次
"""
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Try to import GPU monitoring library
try:
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
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# Try to import CPU/memory monitoring library
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


class SystemProfiler:
    """系统资源监控器 - 在后台线程中定期采样"""
    
    def __init__(
        self,
        output_file: str = "system_metrics.json",
        sample_interval: float = 0.5,
        verbose: bool = False
    ):
        """
        初始化系统资源监控器
        
        Args:
            output_file: 输出 JSON 文件路径
            sample_interval: 采样间隔（秒）
            verbose: 是否打印详细日志
        """
        self.output_file = Path(output_file)
        self.sample_interval = sample_interval
        self.verbose = verbose
        
        self._samples: list[dict[str, Any]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # GPU monitoring
        self._nvml_initialized = False
        self._nvml_handles: list[Any] = []
        self._gpu_names: list[str] = []
        
        # Initialize
        self._init_nvml()
        self._log(f"System Profiler initialized. Output: {self.output_file}")
        self._log(f"NVML available: {NVML_AVAILABLE}, PSUTIL available: {PSUTIL_AVAILABLE}")
    
    def _log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(f"[SystemProfiler] {message}")
    
    def _init_nvml(self):
        """初始化 NVML (NVIDIA Management Library)"""
        if not NVML_AVAILABLE:
            return
        
        try:
            nvmlInit()
            self._nvml_initialized = True
            count = nvmlDeviceGetCount()
            
            for idx in range(count):
                handle = nvmlDeviceGetHandleByIndex(idx)
                self._nvml_handles.append(handle)
                
                try:
                    name = nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    self._gpu_names.append(name)
                except Exception:
                    self._gpu_names.append(f"GPU-{idx}")
            
            self._log(f"NVML initialized. Found {count} GPU(s)")
        except Exception as e:
            self._log(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False
    
    def _shutdown_nvml(self):
        """关闭 NVML"""
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except Exception:
                pass
            finally:
                self._nvml_initialized = False
                self._nvml_handles = []
                self._gpu_names = []
    
    def start(self):
        """启动监控"""
        if self._monitor_thread is not None:
            self._log("Monitor already running")
            return
        
        self._stop_event.clear()
        self._samples = []
        self._monitor_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._monitor_thread.start()
        self._log("Monitor started")
    
    def stop(self):
        """停止监控并保存数据"""
        if self._monitor_thread is None:
            self._log("Monitor not running")
            return
        
        self._stop_event.set()
        self._monitor_thread.join()
        self._monitor_thread = None
        
        self._save_samples()
        self._log(f"Monitor stopped. Saved {len(self._samples)} samples")
    
    def _poll_loop(self):
        """监控循环 - 在后台线程中运行"""
        while not self._stop_event.is_set():
            snapshot = self._collect_snapshot()
            if snapshot is not None:
                with self._lock:
                    self._samples.append(snapshot)
            
            time.sleep(max(self.sample_interval, 0.05))
    
    def _collect_snapshot(self) -> Optional[dict[str, Any]]:
        """收集一次系统资源快照"""
        timestamp = time.time()
        snapshot = {
            "timestamp": timestamp,
        }
        
        # Collect GPU metrics
        if self._nvml_initialized and self._nvml_handles:
            gpus: list[dict[str, Any]] = []
            for idx, handle in enumerate(self._nvml_handles):
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    
                    try:
                        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = None
                    
                    gpus.append({
                        "index": idx,
                        "name": self._gpu_names[idx] if idx < len(self._gpu_names) else f"GPU-{idx}",
                        "sm_util": getattr(util, "gpu", None),
                        "mem_util": getattr(util, "memory", None),
                        "mem_used": getattr(mem, "used", None),
                        "mem_total": getattr(mem, "total", None),
                        "temperature": temp,
                    })
                except Exception:
                    continue
            
            if gpus:
                snapshot["gpus"] = gpus
        
        # Collect CPU and memory metrics
        if PSUTIL_AVAILABLE:
            try:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory
                virtual_mem = psutil.virtual_memory()
                
                # Disk
                disk_usage = psutil.disk_usage('/')
                
                snapshot["cpu"] = {
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                }
                
                snapshot["memory"] = {
                    "mem_used": virtual_mem.used,
                    "mem_total": virtual_mem.total,
                    "mem_available": virtual_mem.available,
                    "mem_percent": virtual_mem.percent,
                }
                
                snapshot["disk"] = {
                    "disk_used": disk_usage.used,
                    "disk_total": disk_usage.total,
                    "disk_free": disk_usage.free,
                    "disk_percent": disk_usage.percent,
                }
            except Exception:
                pass
        
        # Return snapshot only if we have at least some metrics
        if len(snapshot) > 1:  # More than just timestamp
            return snapshot
        return None
    
    def _save_samples(self):
        """保存采样数据到 JSON 文件"""
        if not self._samples:
            self._log("No samples to save")
            return
        
        try:
            # Ensure parent directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON
            with self.output_file.open("w", encoding="utf-8") as f:
                json.dump(self._samples, f, ensure_ascii=False, indent=2)
            
            self._log(f"Saved {len(self._samples)} samples to {self.output_file}")
        except Exception as e:
            self._log(f"Failed to save samples: {e}")
    
    def get_summary(self) -> dict[str, Any]:
        """获取统计摘要"""
        if not self._samples:
            return {"error": "No samples collected"}
        
        summary = {
            "total_samples": len(self._samples),
            "duration_seconds": self._samples[-1]["timestamp"] - self._samples[0]["timestamp"],
        }
        
        # GPU summary
        if "gpus" in self._samples[0]:
            gpu_utils = [s["gpus"][0]["sm_util"] for s in self._samples if "gpus" in s and s["gpus"]]
            gpu_mem_utils = [s["gpus"][0]["mem_util"] for s in self._samples if "gpus" in s and s["gpus"]]
            
            if gpu_utils:
                summary["gpu"] = {
                    "avg_utilization": sum(gpu_utils) / len(gpu_utils),
                    "max_utilization": max(gpu_utils),
                    "min_utilization": min(gpu_utils),
                    "avg_mem_utilization": sum(gpu_mem_utils) / len(gpu_mem_utils),
                    "max_mem_utilization": max(gpu_mem_utils),
                }
        
        # CPU summary
        if "cpu" in self._samples[0]:
            cpu_percents = [s["cpu"]["cpu_percent"] for s in self._samples if "cpu" in s]
            
            if cpu_percents:
                summary["cpu"] = {
                    "avg_percent": sum(cpu_percents) / len(cpu_percents),
                    "max_percent": max(cpu_percents),
                    "min_percent": min(cpu_percents),
                }
        
        # Memory summary
        if "memory" in self._samples[0]:
            mem_percents = [s["memory"]["mem_percent"] for s in self._samples if "memory" in s]
            
            if mem_percents:
                summary["memory"] = {
                    "avg_percent": sum(mem_percents) / len(mem_percents),
                    "max_percent": max(mem_percents),
                    "min_percent": min(mem_percents),
                }
        
        return summary
    
    def __del__(self):
        """清理资源"""
        self.stop()
        self._shutdown_nvml()


def main():
    """命令行测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="System Resource Profiler")
    parser.add_argument(
        "--output",
        default="system_metrics.json",
        help="Output JSON file (default: system_metrics.json)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Monitoring duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs"
    )
    
    args = parser.parse_args()
    
    profiler = SystemProfiler(
        output_file=args.output,
        sample_interval=args.interval,
        verbose=args.verbose
    )
    
    print(f"Starting system profiler for {args.duration} seconds...")
    print(f"Sampling every {args.interval} seconds")
    print(f"Output will be saved to: {args.output}")
    print()
    
    profiler.start()
    time.sleep(args.duration)
    profiler.stop()
    
    # Print summary
    summary = profiler.get_summary()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print()


if __name__ == "__main__":
    main()
