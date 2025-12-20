#!/usr/bin/env python3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


VLLM_CMD = [
    "vllm",
    "serve",
    "/mnt/shared/models/GLM-4.6-FP8",
    "--kv_cache_dtype",
    "fp8",
    "--port",
    "8100",
    "--tensor-parallel-size",
    "8",
]


def _wait_vllm_healthy(proc: subprocess.Popen) -> None:
    urls = (
        "http://127.0.0.1:8100/health",
        "http://127.0.0.1:8100/v1/models",
    )
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        for url in urls:
            try:
                with urllib.request.urlopen(url) as resp:
                    if resp.status == 200:
                        print(f"vLLM healthy: {url}")
                        return
            except Exception:
                pass
        time.sleep(1)


def main() -> int:
    vllm_proc = subprocess.Popen(VLLM_CMD)
    try:
        _wait_vllm_healthy(vllm_proc)
        openhands_dir = Path(__file__).resolve().parent / "OpenHands"
        cmd = [
            sys.executable,
            "-m",
            "evaluation.benchmarks.swe_bench.run_infer",
            "--llm-config",
            "vllm_local",
            "--agent-cls",
            "CodeActAgent",
            "--dataset",
            "princeton-nlp/SWE-bench_Lite",
            "--split",
            "test",
            "--max-iterations",
            "100",
            "--eval-num-workers",
            "128",
            "--eval-output-dir",
            "./outputs",
        ]
        subprocess.run(cmd, cwd=str(openhands_dir), check=True)
        return 0
    finally:
        if vllm_proc.poll() is None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
