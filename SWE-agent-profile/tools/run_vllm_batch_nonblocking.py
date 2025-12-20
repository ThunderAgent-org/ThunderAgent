#!/usr/bin/env python3
"""Run local vLLM batches without step synchronization."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from rich.live import Live

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import run_vllm_batch as blocking
from run_batch_nonblock import RunBatchConfig, RunBatch
from sweagent.utils.log import get_logger

CLEANUP_LOGGER = get_logger("docker-cleanup", emoji="🧹")
_CLEANUP_PATCH_APPLIED = False
HEALTH_CHECK_LOGGER = get_logger("vllm-health", emoji="🏥")


def _derive_container_name_patterns(instance_id: str) -> set[str]:
    candidate = instance_id.strip()
    if not candidate:
        return set()
    patterns = {candidate}
    sanitized = candidate.replace("/", "_")
    patterns.add(sanitized)
    patterns.add(candidate.replace("/", "-"))
    patterns.add(sanitized.replace("__", "_"))
    patterns.add(sanitized.replace("__", "-"))
    patterns.add(candidate.lower())
    patterns.add(sanitized.lower())
    return {pattern for pattern in patterns if pattern}


def _list_all_containers(*, logger = CLEANUP_LOGGER) -> list[dict[str, Any]]:
    if shutil.which("docker") is None:
        return []
    try:
        completed = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{json .}}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("Docker executable not found while listing containers; skipping cleanup step")
        return []
    if completed.returncode != 0:
        if completed.stderr.strip():
            logger.debug("Failed to list docker containers: %s", completed.stderr.strip())
        return []
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Could not parse docker ps output line: %s", line)
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def _find_containers_for_instance(instance_id: str, *, logger = CLEANUP_LOGGER) -> list[dict[str, Any]]:
    patterns = _derive_container_name_patterns(instance_id)
    if not patterns:
        return []
    matches: list[dict[str, Any]] = []
    for record in _list_all_containers(logger=logger):
        name = str(record.get("Names") or "")
        if any(pattern in name for pattern in patterns):
            matches.append(record)
    return matches


def _run_docker_command(command: list[str], *, ignore_errors: bool = False, logger = CLEANUP_LOGGER) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        logger.warning("Docker executable not found while running cleanup command; skipping")
        return
    cmd_display = " ".join(command)
    if completed.stdout.strip():
        logger.debug("Docker command output [%s]: %s", cmd_display, completed.stdout.strip())
    if completed.returncode != 0:
        message = completed.stderr.strip()
        if ignore_errors:
            if message:
                logger.debug("Docker command failed (ignored) [%s]: %s", cmd_display, message)
        else:
            logger.warning("Docker command failed [%s] (code %s): %s", cmd_display, completed.returncode, message)


def _collect_instance_images(instance: Any, containers: list[dict[str, Any]]) -> set[str]:
    images: set[str] = set()
    for record in containers:
        image = str(record.get("Image") or "").strip()
        if image and image not in {"<none>", "<none>:<none>"}:
            images.add(image)
    deployment = getattr(getattr(instance, "env", None), "deployment", None)
    image_name = getattr(deployment, "image", None) if deployment is not None else None
    if isinstance(image_name, str) and image_name:
        images.add(image_name)
    return images


def _cleanup_docker_artifacts(run: Any, instance: Any) -> None:
    if shutil.which("docker") is None:
        return
    problem_statement = getattr(instance, "problem_statement", None)
    instance_id = getattr(problem_statement, "id", None)
    if not instance_id:
        return
    logger = getattr(run, "logger", CLEANUP_LOGGER)
    containers = _find_containers_for_instance(instance_id, logger=logger)
    images = _collect_instance_images(instance, containers)
    if not containers and not images:
        logger.debug("No docker artifacts detected for instance %s", instance_id)
        return
    logger.info(
        "Cleaning docker artifacts for instance %s (%d containers, %d images)",
        instance_id,
        len(containers),
        len(images),
    )
    for record in containers:
        target = str(record.get("ID") or record.get("Names") or "").strip()
        if not target:
            continue
        _run_docker_command(["docker", "rm", "-f", target], ignore_errors=True, logger=logger)
    for image in sorted(images):
        _run_docker_command(["docker", "image", "rm", image], ignore_errors=True, logger=logger)


def _install_instance_cleanup_patch() -> None:
    global _CLEANUP_PATCH_APPLIED
    if _CLEANUP_PATCH_APPLIED:
        return

    original_run_instance = RunBatch.run_instance

    def run_instance_with_cleanup(self, instance, *args, **kwargs):  # type: ignore[override]
        try:
            return original_run_instance(self, instance, *args, **kwargs)
        finally:
            try:
                _cleanup_docker_artifacts(self, instance)
            except Exception as exc:  # noqa: BLE001
                instance_id = getattr(getattr(instance, "problem_statement", None), "id", "<unknown>")
                CLEANUP_LOGGER.warning(
                    "Encountered error while cleaning docker artifacts for %s: %s",
                    instance_id,
                    exc,
                )

    RunBatch.run_instance = run_instance_with_cleanup  # type: ignore[assignment]
    _CLEANUP_PATCH_APPLIED = True


_install_instance_cleanup_patch()


def check_server_health(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if server is responsive on the given host:port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def health_monitor_thread(
    host: str,
    port: int,
    server_proc: subprocess.Popen,
    server_cmd: list[str],
    server_env: dict[str, str] | None,
    stop_event: threading.Event,
    restart_callback,
    check_interval: int = 30,
    consecutive_failures: int = 3
):
    """Monitor server health and restart if necessary
    
    Args:
        host: Server host
        port: Server port
        server_proc: Current server process
        server_cmd: Command to restart server
        server_env: Environment variables for server
        stop_event: Event to signal thread to stop
        restart_callback: Function to call with new process when restarting
        check_interval: Seconds between health checks
        consecutive_failures: Number of consecutive failures before restart
    """
    failure_count = 0
    restart_count = 0
    max_restarts = 5
    
    HEALTH_CHECK_LOGGER.info(f"Health monitor started (checking every {check_interval}s)")
    
    while not stop_event.is_set():
        # Wait for check_interval or until stop_event is set
        if stop_event.wait(timeout=check_interval):
            break
        
        # Check if process is still running
        if server_proc.poll() is not None:
            HEALTH_CHECK_LOGGER.warning(f"Server process died (exit code: {server_proc.returncode})")
            failure_count = consecutive_failures  # Immediate restart
        # Check if port is responsive
        elif not check_server_health(host, port, timeout=5.0):
            failure_count += 1
            HEALTH_CHECK_LOGGER.warning(
                f"Server health check failed ({failure_count}/{consecutive_failures})"
            )
        else:
            # Health check passed
            if failure_count > 0:
                HEALTH_CHECK_LOGGER.info("Server recovered")
            failure_count = 0
            continue
        
        # Restart if we've reached consecutive failures
        if failure_count >= consecutive_failures:
            if restart_count >= max_restarts:
                HEALTH_CHECK_LOGGER.error(
                    f"Maximum restart attempts ({max_restarts}) reached. Giving up."
                )
                break
            
            restart_count += 1
            HEALTH_CHECK_LOGGER.warning(
                f"Attempting to restart server (attempt {restart_count}/{max_restarts})..."
            )
            
            # Kill old process
            try:
                if server_proc.poll() is None:
                    server_proc.terminate()
                    time.sleep(2)
                    if server_proc.poll() is None:
                        server_proc.kill()
                        server_proc.wait(timeout=5)
            except Exception as e:
                HEALTH_CHECK_LOGGER.error(f"Error killing old process: {e}")
            
            # Start new process
            try:
                new_proc = subprocess.Popen(server_cmd, env=server_env)
                HEALTH_CHECK_LOGGER.info(f"New server process started (PID: {new_proc.pid})")
                
                # Wait for server to be ready
                max_wait = 120
                waited = 0
                while waited < max_wait and not stop_event.is_set():
                    if check_server_health(host, port, timeout=2.0):
                        HEALTH_CHECK_LOGGER.info(
                            f"Server successfully restarted and is responsive"
                        )
                        restart_callback(new_proc)
                        failure_count = 0
                        break
                    time.sleep(5)
                    waited += 5
                else:
                    if not stop_event.is_set():
                        HEALTH_CHECK_LOGGER.error(
                            f"Server failed to become responsive after {max_wait}s"
                        )
                        failure_count = 0  # Reset to try again later
            except Exception as e:
                HEALTH_CHECK_LOGGER.error(f"Failed to restart server: {e}")
                failure_count = 0  # Reset to try again later
    
    HEALTH_CHECK_LOGGER.info("Health monitor stopped")


def build_thunderreact_command(
    server_cfg: dict, 
    thunderreact_path: str, 
    proxy_port: int, 
    enable_logging: bool = False,
    enable_system_profiling: bool = False,
    profiling_interval: float = 0.5
) -> list[str]:
    """Build ThunderReact command that wraps vLLM"""
    cmd = [
        sys.executable,
        thunderreact_path,
        "--auto-start-vllm",
        "--port", str(proxy_port),
        "--vllm-port", str(server_cfg.get("port", 8000)),
        "--log-dir", "./thunderreact_logs",
        # Remove --verbose to avoid console spam
    ]
    
    # Add --enable-logging if requested
    if enable_logging:
        cmd.append("--enable-logging")
    
    # Add system profiling parameters if requested
    if enable_system_profiling:
        cmd.append("--enable-system-profiling")
        cmd.extend(["--profiling-interval", str(profiling_interval)])
        # Output file will be in thunderreact_logs/system_metrics.json by default
    
    # Add all vLLM parameters
    if "model" in server_cfg:
        cmd.extend(["--model", str(server_cfg["model"])])
    if "tokenizer" in server_cfg:
        cmd.extend(["--tokenizer", str(server_cfg["tokenizer"])])
    if "host" in server_cfg:
        cmd.extend(["--host", str(server_cfg["host"])])
    if "gpu_memory_utilization" in server_cfg:
        cmd.extend(["--gpu-memory-utilization", str(server_cfg["gpu_memory_utilization"])])
    if "dtype" in server_cfg:
        cmd.extend(["--dtype", str(server_cfg["dtype"])])
    if "max_model_len" in server_cfg:
        cmd.extend(["--max-model-len", str(server_cfg["max_model_len"])])
    
    # Add extra_args from config
    if "extra_args" in server_cfg:
        for arg in server_cfg["extra_args"]:
            cmd.append(str(arg))
    
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start vLLM server and run sweagent batches without step synchronization.",
    )
    parser.add_argument(
        "--config-file",
        default="config/vllm_batch.yaml",
        help="Path to the YAML config describing vLLM server and batch options.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--only-runs",
        nargs="*",
        help="Only execute runs whose label matches one of the provided names.",
    )
    parser.add_argument(
        "--use-thunderreact",
        action="store_true",
        help="Use ThunderReact proxy instead of direct vLLM (enables request/response logging).",
    )
    parser.add_argument(
        "--thunderreact-path",
        default="/root/workspace/MultiagentSystem/thunderreact/simple_proxy.py",
        help="Path to ThunderReact simple_proxy.py script.",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=9000,
        help="ThunderReact proxy port (only used with --use-thunderreact).",
    )
    parser.add_argument(
        "--enable-proxy-logging",
        action="store_true",
        help="Enable ThunderReact request/response logging to files.",
    )
    parser.add_argument(
        "--enable-system-profiling",
        action="store_true",
        help="Enable system-wide resource profiling (GPU/CPU/Memory/Disk).",
    )
    parser.add_argument(
        "--profiling-interval",
        type=float,
        default=0.5,
        help="System profiling sampling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=30,
        help="Server health check interval in seconds (default: 30). Set to 0 to disable.",
    )
    parser.add_argument(
        "--disable-health-check",
        action="store_true",
        help="Disable automatic server health monitoring and restart.",
    )
    return parser.parse_args()


def _apply_overrides(config_path: str, args_list: Sequence[str]) -> tuple[dict[str, Any], str]:
    base_cfg = blocking.load_config(config_path)
    applied_cfg, suffix = blocking._apply_cli_overrides(base_cfg, args_list)
    base_suffix = applied_cfg.get("suffix", "")
    if suffix is not None:
        if base_suffix and suffix:
            base_suffix = f"{base_suffix}{suffix}"
        else:
            base_suffix = suffix or ""
    if base_suffix:
        applied_cfg["suffix"] = base_suffix
    elif "suffix" in applied_cfg:
        applied_cfg.pop("suffix", None)
    return applied_cfg, base_suffix


def _run_nonblocking(config_path: str, args_list: Sequence[str], label: str | None, env: Mapping[str, Any] | None) -> None:
    applied_cfg, suffix = _apply_overrides(config_path, args_list)
    if label:
        print(f"[debug] configuration for {label}: {applied_cfg.get('agent', {}).get('model')}")
    cfg = RunBatchConfig.model_validate(applied_cfg)
    if suffix:
        cfg.suffix = suffix
    cfg._config_files = [config_path]  # type: ignore[attr-defined]

    env_updates = {str(k): str(v) for k, v in (env or {}).items()}
    old_env: dict[str, str | None] = {}
    for key, value in env_updates.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        runner = RunBatch.from_config(cfg)
        runner.main()
    finally:
        for key, previous in old_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def main() -> int:
    args = parse_args()
    raw_config = blocking.load_config(args.config_file)
    server_cfg = blocking.ensure_mapping(raw_config.get("vllm"), "vllm")
    batch_cfg = blocking.ensure_mapping(raw_config.get("batch"), "batch")

    sweagent_config = str(batch_cfg.get("sweagent_config", "config/vllm_local.yaml"))
    batch_args = [str(arg) for arg in batch_cfg.get("args", []) or []]
    runs = batch_cfg.get("runs")
    if runs is not None and not isinstance(runs, Sequence):
        raise ValueError("batch.runs must be a sequence if provided")

    selected_runs = set(args.only_runs or [])

    # Build server command (vLLM or ThunderReact)
    if args.use_thunderreact:
        server_cmd = build_thunderreact_command(
            server_cfg, 
            args.thunderreact_path, 
            args.proxy_port, 
            args.enable_proxy_logging,
            args.enable_system_profiling,
            args.profiling_interval
        )
        print("Starting ThunderReact proxy (wrapping vLLM):", " ".join(server_cmd))
        # Update config to point to proxy port
        connect_host = str(server_cfg.get("connect_host", server_cfg.get("host", "127.0.0.1")))
        port = args.proxy_port  # Use proxy port instead of vLLM port
    else:
        server_cmd = blocking.build_vllm_command(server_cfg)
        print("Starting vLLM server:", " ".join(server_cmd))
        connect_host = str(server_cfg.get("connect_host", server_cfg.get("host", "127.0.0.1")))
        port = int(server_cfg.get("port", 8000))

    if args.dry_run:
        print("[dry-run] would wait for server and run sweagent batch (non-blocking)")
        seen: set[str] = set()
        if runs:
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = blocking._extract_run(run_cfg, idx)
                if selected_runs and label not in selected_runs:
                    continue
                seen.add(label)
                combined_args = list(batch_args) + run_args
                combined_env = blocking._merge_two_envs(batch_cfg.get("env"), run_env)
                blocking.run_batch(sweagent_config, combined_args, env=combined_env, dry_run=True, label=label)
        else:
            blocking.run_batch(sweagent_config, batch_args, env=batch_cfg.get("env"), dry_run=True)
        if selected_runs and selected_runs - seen:
            missing = ", ".join(sorted(selected_runs - seen))
            raise ValueError(f"Requested runs not found in config: {missing}")
        return 0

    server_proc = subprocess.Popen(server_cmd, env=blocking._merge_env(server_cfg.get("env")))
    server_proc_lock = threading.Lock()
    health_monitor_stop = threading.Event()
    health_monitor = None
    
    def update_server_proc(new_proc: subprocess.Popen):
        """Callback to update server_proc reference when restarted"""
        nonlocal server_proc
        with server_proc_lock:
            server_proc = new_proc
    
    try:
        timeout = int(server_cfg.get("wait_timeout", raw_config.get("wait_timeout", 180)))
        blocking.wait_for_server(connect_host, port, timeout)
        
        # Start health monitor thread (unless disabled)
        if not args.disable_health_check and args.health_check_interval > 0:
            health_check_interval = args.health_check_interval
            health_monitor = threading.Thread(
                target=health_monitor_thread,
                args=(
                    connect_host,
                    port,
                    server_proc,
                    server_cmd,
                    blocking._merge_env(server_cfg.get("env")),
                    health_monitor_stop,
                    update_server_proc,
                    health_check_interval,
                    3  # consecutive failures before restart
                ),
                daemon=True,
                name="vllm-health-monitor"
            )
            health_monitor.start()
            HEALTH_CHECK_LOGGER.info(
                f"Health monitoring enabled (interval: {health_check_interval}s, max restarts: 5)"
            )
        else:
            HEALTH_CHECK_LOGGER.info("Health monitoring disabled")
        
        if args.use_thunderreact:
            print(f"ThunderReact proxy is ready on port {port}. Running sweagent batch (non-blocking)...")
            if args.enable_proxy_logging:
                print(f"📝 Request/response logs: ./thunderreact_logs/{{requests,responses}}.jsonl")
            else:
                print(f"📝 Request/response logging disabled (add --enable-proxy-logging to enable)")
            
            if args.enable_system_profiling:
                print(f"📊 System profiling enabled:")
                print(f"   Output: ./thunderreact_logs/system_metrics.json")
                print(f"   Interval: {args.profiling_interval}s")
            else:
                print(f"📊 System profiling disabled (add --enable-system-profiling to enable)")
        else:
            print("vLLM server is ready. Running sweagent batch (non-blocking)...")

        exit_code = 0
        seen: set[str] = set()
        if runs:
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = blocking._extract_run(run_cfg, idx)
                if selected_runs and label not in selected_runs:
                    continue
                seen.add(label)
                combined_args = list(batch_args) + run_args
                combined_env = blocking._merge_two_envs(batch_cfg.get("env"), run_env)
                blocking._prepare_run_cleanup(sweagent_config, combined_args, dry_run=False, label=label)
                try:
                    _run_nonblocking(sweagent_config, combined_args, label, combined_env)
                except Exception as exc:  # noqa: BLE001
                    print(f"Batch '{label}' failed: {exc}", file=sys.stderr)
                    exit_code = 1
                    break
        else:
            blocking._prepare_run_cleanup(sweagent_config, batch_args, dry_run=False)
            try:
                _run_nonblocking(sweagent_config, batch_args, None, batch_cfg.get("env"))
            except Exception as exc:  # noqa: BLE001
                print(f"Batch run failed: {exc}", file=sys.stderr)
                exit_code = 1
        if selected_runs and selected_runs - seen:
            missing = ", ".join(sorted(selected_runs - seen))
            raise ValueError(f"Requested runs not found in config: {missing}")
        return exit_code
    finally:
        # Stop health monitor
        if health_monitor and health_monitor.is_alive():
            HEALTH_CHECK_LOGGER.info("Stopping health monitor...")
            health_monitor_stop.set()
            health_monitor.join(timeout=5)
        
        # Stop server
        if args.use_thunderreact:
            print("Stopping ThunderReact proxy (and vLLM)...")
        else:
            print("Stopping vLLM server...")
        with server_proc_lock:
            blocking.terminate_process(server_proc)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())

