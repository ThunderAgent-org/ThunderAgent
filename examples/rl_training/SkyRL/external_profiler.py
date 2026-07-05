#!/usr/bin/env python3
"""
External vLLM metrics profiler for SkyRL training.

Collects metrics from two sources:
1. vLLM engine stat logs from Ray worker log files (KV cache, preemptions, throughput)
2. HTTP /metrics endpoint (in-flight requests per engine)

Runs inside the Docker container via srun --overlap.

Usage: python3 external_profiler.py [--output-dir /scratch/profiler_data] [--poll-interval 5]
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

# Pattern to match vLLM LoggingStatLogger output
# "Engine 000: Avg prompt throughput: 12.3 tokens/s, Avg generation throughput: 45.6 tokens/s,
#  Running: 5 reqs, Waiting: 0 reqs, [Preemptions: N,] GPU KV cache usage: 78.9%,
#  Prefix cache hit rate: 12.3%"
VLLM_STATS_PATTERN = re.compile(
    r"Engine (\d+): "
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, "
    r"Waiting: (\d+) reqs"
    r"(?:, Preemptions: (\d+))?"
    r"(?:, GPU KV cache usage: ([\d.]+)%)?"
    r"(?:, Prefix cache hit rate: ([\d.]+)%)?"
)

# Pattern for timestamps in Ray worker logs (vLLM uses loguru-style timestamps)
TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+)")


def find_ray_log_dir():
    """Find the Ray session log directory."""
    candidates = [
        "/tmp/ray/session_latest/logs/",
        "/tmp/ray/logs/",
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    # Fallback: find any ray session
    ray_dir = "/tmp/ray/"
    if os.path.isdir(ray_dir):
        for entry in sorted(os.listdir(ray_dir), reverse=True):
            logs_path = os.path.join(ray_dir, entry, "logs")
            if os.path.isdir(logs_path):
                return logs_path
    return None


def scan_worker_logs(log_dir, last_positions=None):
    """Scan Ray worker log files for vLLM stats lines."""
    if last_positions is None:
        last_positions = {}

    results = []
    # vLLM uses Python logging which goes to stderr in Ray
    worker_files = sorted(glob.glob(os.path.join(log_dir, "worker-*.err")))
    worker_files += sorted(glob.glob(os.path.join(log_dir, "worker-*.out")))

    for filepath in worker_files:
        try:
            file_size = os.path.getsize(filepath)
            last_pos = last_positions.get(filepath, 0)

            if file_size <= last_pos:
                continue

            with open(filepath, "r", errors="ignore") as f:
                f.seek(last_pos)
                for line in f:
                    match = VLLM_STATS_PATTERN.search(line)
                    if match:
                        # Try to extract timestamp from the log line
                        ts_match = TIMESTAMP_PATTERN.search(line)
                        log_ts = ts_match.group(1) if ts_match else None

                        entry = {
                            "timestamp": time.time(),
                            "log_timestamp": log_ts,
                            "source": "ray_log",
                            "log_file": os.path.basename(filepath),
                            "engine_id": int(match.group(1)),
                            "prompt_throughput_tps": float(match.group(2)),
                            "generation_throughput_tps": float(match.group(3)),
                            "num_running_reqs": int(match.group(4)),
                            "num_waiting_reqs": int(match.group(5)),
                        }
                        if match.group(6) is not None:
                            entry["num_preemptions"] = int(match.group(6))
                        if match.group(7) is not None:
                            entry["kv_cache_usage_pct"] = float(match.group(7))
                        if match.group(8) is not None:
                            entry["prefix_cache_hit_rate_pct"] = float(match.group(8))
                        results.append(entry)

                last_positions[filepath] = f.tell()
        except Exception as e:
            pass  # Silently skip unreadable files

    return results, last_positions


def poll_http_metrics(url, timeout=5):
    """Poll the SkyRL HTTP /metrics endpoint for in-flight request counts."""
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            data["source"] = "http_endpoint"
            data["poll_time"] = time.time()
            return data
    except Exception as e:
        return {"source": "http_endpoint", "error": str(e), "poll_time": time.time()}
    return None


def count_docker_containers():
    """Count running Docker containers (if docker is available)."""
    try:
        import subprocess

        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}", "--filter", "name=minisweagent"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            containers = [c for c in result.stdout.strip().split("\n") if c]
            return len(containers)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="External vLLM metrics profiler for SkyRL")
    parser.add_argument("--output-dir", default="/scratch/profiler_data", help="Output directory for metrics")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--http-url", default="http://127.0.0.1:8001/metrics", help="SkyRL HTTP metrics URL")
    parser.add_argument("--docker-interval", type=int, default=30, help="Docker container count interval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = int(time.time())
    output_file = os.path.join(args.output_dir, f"profiler_{start_time}.jsonl")

    log_dir = find_ray_log_dir()
    print(f"=== External vLLM Profiler ===", flush=True)
    print(f"Start time: {datetime.now().isoformat()}", flush=True)
    print(f"Ray log directory: {log_dir}", flush=True)
    print(f"Output file: {output_file}", flush=True)
    print(f"Poll interval: {args.poll_interval}s", flush=True)
    print(f"HTTP endpoint: {args.http_url}", flush=True)

    if log_dir:
        worker_count = len(glob.glob(os.path.join(log_dir, "worker-*.err")))
        print(f"Worker log files found: {worker_count}", flush=True)
    else:
        print("WARNING: No Ray log directory found!", flush=True)

    last_positions = {}
    total_log_entries = 0
    total_http_entries = 0
    cycle = 0
    last_docker_check = 0

    with open(output_file, "a") as out_f:
        # Write header metadata
        header = {
            "type": "header",
            "start_time": start_time,
            "ray_log_dir": log_dir,
            "http_url": args.http_url,
            "poll_interval": args.poll_interval,
        }
        out_f.write(json.dumps(header) + "\n")
        out_f.flush()

        while True:
            try:
                now = time.time()

                # 1. Scan Ray worker logs for vLLM engine metrics
                if log_dir:
                    log_results, last_positions = scan_worker_logs(log_dir, last_positions)
                    for entry in log_results:
                        out_f.write(json.dumps(entry) + "\n")
                        total_log_entries += 1

                    # Report new vLLM log entries
                    if log_results:
                        latest = log_results[-1]
                        kv_str = f"KV={latest.get('kv_cache_usage_pct', '?')}%" if 'kv_cache_usage_pct' in latest else "KV=N/A"
                        preempt_str = f"preempt={latest.get('num_preemptions', 0)}" if 'num_preemptions' in latest else ""
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"vLLM engine {latest['engine_id']}: "
                            f"{kv_str}, "
                            f"running={latest['num_running_reqs']}, "
                            f"waiting={latest['num_waiting_reqs']}, "
                            f"gen_tps={latest['generation_throughput_tps']:.1f} "
                            f"{preempt_str} "
                            f"(+{len(log_results)} entries, total={total_log_entries})",
                            flush=True,
                        )

                # 2. Poll HTTP /metrics endpoint
                http_data = poll_http_metrics(args.http_url)
                if http_data and "error" not in http_data:
                    out_f.write(json.dumps(http_data) + "\n")
                    total_http_entries += 1

                    # Print compact per-engine KV cache summary to stderr
                    # (visible in sbatch .err / terminal even when stdout is redirected)
                    engines = http_data.get("engines", [])
                    if engines:
                        parts = []
                        total_running = 0
                        total_preempt = 0
                        for e in engines:
                            kv = e.get("kv_cache_usage_pct", -1)
                            ru = e.get("num_running_reqs", 0)
                            pr = e.get("num_cumulative_preemption", 0)
                            total_running += ru
                            total_preempt += pr
                            parts.append(f"e{e.get('engine_id', '?')}:{kv:5.1f}%({ru}r)")
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"running={total_running} preempt={total_preempt} | "
                            f"{' '.join(parts)}",
                            file=sys.stderr,
                            flush=True,
                        )

                # 3. Periodically count Docker containers
                if now - last_docker_check > args.docker_interval:
                    container_count = count_docker_containers()
                    if container_count is not None:
                        docker_entry = {
                            "source": "docker",
                            "timestamp": now,
                            "minisweagent_containers": container_count,
                        }
                        out_f.write(json.dumps(docker_entry) + "\n")
                        if cycle % 12 == 0:  # Print every ~minute
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Docker containers: {container_count}", flush=True)
                    last_docker_check = now

                out_f.flush()
                cycle += 1

                # Periodic summary
                if cycle % 60 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Summary: {total_log_entries} vLLM log entries, "
                        f"{total_http_entries} HTTP polls, "
                        f"elapsed={elapsed/60:.1f}min",
                        flush=True,
                    )

                time.sleep(args.poll_interval)

            except KeyboardInterrupt:
                print(f"\nStopping profiler. Total: {total_log_entries} log entries, {total_http_entries} HTTP polls", flush=True)
                break
            except Exception as e:
                print(f"Error in profiler loop: {e}", file=sys.stderr, flush=True)
                time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
