#!/usr/bin/env python3
import subprocess
import sys
import time
import urllib.request
import os
import shutil
import threading
from pathlib import Path


VLLM_CMD = [
    "vllm",
    "serve",
    "/mnt/shared/models/GLM-4.6-FP8",
    # "--kv_cache_dtype",
    # "fp8",
    "--port",
    "8100",
    "--tensor-parallel-size",
    "8",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "glm45",
    "--reasoning-parser",
    "glm45",
]


def _wait_vllm_healthy(proc: subprocess.Popen) -> None:
    """等待 vLLM 完全启动，包括模型加载完成"""
    import json
    
    health_url = "http://127.0.0.1:8100/health"
    models_url = "http://127.0.0.1:8100/v1/models"
    
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        
        try:
            # 先检查 health 端点
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status != 200:
                    time.sleep(1)
                    continue
            
            # 再检查 models 端点，确认模型已加载
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = data.get("data", [])
                    if models:
                        print(f"vLLM ready! Loaded models: {[m.get('id') for m in models]}")
                        return
                    else:
                        print("vLLM health OK, but no models loaded yet...")
        except Exception as e:
            pass
        
        time.sleep(1)


def monitor_system(stop_event):
    log_file = "system_monitor.csv"
    
    def get_cpu():
        with open('/proc/stat') as f:
            v = [float(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4] # total, idle

    last_tot, last_idl = get_cpu()
    f = open(log_file, 'w')
    header_set = False

    while not stop_event.is_set():
        try:
            gpus = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], 
                text=True).strip().split('\n')
            g_vals = [x.strip() for line in gpus for x in line.split(',')] if gpus[0] else []
        except: g_vals = []

        with open('/proc/meminfo') as m:
            md = {l.split(':')[0]: int(l.split()[1]) for l in m}
        mem = (md['MemTotal'] - md.get('MemAvailable', md['MemFree'])) / 1024**2
        
        load = os.getloadavg()[0]
        tot, idl = get_cpu()
        util = 100 * (1 - (idl - last_idl) / (tot - last_tot)) if (tot - last_tot) > 0 else 0
        last_tot, last_idl = tot, idl
        
        disk = shutil.disk_usage('/').used / 1024**3
        
        if not header_set:
            gh = []
            for i in range(len(g_vals)//2): gh.extend([f"GPU{i}_Mem", f"GPU{i}_Util"])
            f.write(','.join(["Time"] + gh + ["CPU_Mem(GiB)", "Load", "CPU_Util(%)", "Disk_Overlay(GiB)"]) + "\n")
            header_set = True
            
        f.write(f"{time.strftime('%H:%M:%S')},{','.join(g_vals)},{mem:.1f},{load:.2f},{util:.1f},{disk:.1f}\n")
        f.flush()
        time.sleep(2)
    f.close()


def main() -> int:
    vllm_proc = subprocess.Popen(VLLM_CMD)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_system, args=(stop_event,))
    monitor_thread.start()
    try:
        _wait_vllm_healthy(vllm_proc)
        openhands_dir = Path(__file__).resolve().parent / "OpenHands"
        config_file = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "config.toml"
        cmd = [
            sys.executable,
            "-m",
            "evaluation.benchmarks.swe_bench.run_infer",
            "--llm-config",
            "vllm_local",
            "--config-file",
            str(config_file),
            "--agent-cls",
            "CodeActAgent",
            "--dataset",
            "princeton-nlp/SWE-bench_Lite",
            "--split",
            "test",
            "--max-iterations",
            "50",
            "--eval-num-workers",
            "24",
            "--eval-output-dir",
            "./outputs",
        ]
        subprocess.run(cmd, cwd=str(openhands_dir), check=True)
        return 0
    finally:
        stop_event.set()
        monitor_thread.join()
        if vllm_proc.poll() is None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
