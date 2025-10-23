#!/usr/bin/env python3
"""Launch a local vLLM OpenAI-compatible server and run a SWE-agent batch.

All parameters for the vLLM server and the downstream `sweagent run-batch`
invocation are specified in a YAML configuration file so that different
setups can be captured declaratively.
"""

from __future__ import annotations

import argparse
import http.client
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sweagent.run.common import _parse_args_to_nested_dict
from sweagent.utils.serialization import merge_nested_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible server and run `sweagent run-batch` using settings from a YAML file.",
    )
    parser.add_argument(
        "--config-file",
        default="config/vllm_batch.yaml",
        help="Path to YAML config describing vLLM server and run-batch options (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--only-runs",
        nargs="+",
        help="Only execute the runs whose label (from batch.runs[].label) or numeric index matches one of the provided values.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return dict(data)


def ensure_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Section '{name}' must be a mapping")
    return dict(value)


def build_vllm_command(config: Mapping[str, Any]) -> list[str]:
    if "model" not in config:
        raise ValueError("vLLM config must contain 'model'")
    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config["model"]),
        "--host",
        str(config.get("host", "0.0.0.0")),
        "--port",
        str(config.get("port", 8000)),
    ]
    if tokenizer := config.get("tokenizer"):
        cmd.extend(["--tokenizer", str(tokenizer)])
    if tensor_parallel := config.get("tensor_parallel"):
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
    if gpu_mem := config.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(gpu_mem)])
    if dtype := config.get("dtype"):
        cmd.extend(["--dtype", str(dtype)])
    if max_len := config.get("max_model_len"):
        cmd.extend(["--max-model-len", str(max_len)])
    extra_args = config.get("extra_args", []) or []
    for extra in extra_args:
        cmd.append(str(extra))
    return cmd


def wait_for_server(host: str, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            if response.status in {200, 401, 403}:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError(f"vLLM server did not become ready within {timeout}s: {last_error}")


def _merge_env(extra_env: Mapping[str, Any] | None) -> Mapping[str, str]:
    base = os.environ.copy()
    if not extra_env:
        return base
    base.update({str(k): str(v) for k, v in extra_env.items()})
    return base


def _merge_two_envs(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not base and not override:
        return {}
    merged: dict[str, Any] = {}
    if base:
        merged.update({str(k): v for k, v in base.items()})
    if override:
        merged.update({str(k): v for k, v in override.items()})
    return merged


def _extract_run(run_cfg: Any, idx: int) -> tuple[list[str], Mapping[str, Any] | None, str]:
    if isinstance(run_cfg, Mapping):
        args = run_cfg.get("args", [])
        if args and not isinstance(args, Sequence):
            raise ValueError("batch.runs[].args must be a sequence")
        env = run_cfg.get("env")
        label = run_cfg.get("label") or f"run-{idx}"
        return [str(arg) for arg in args], env, label
    if isinstance(run_cfg, Sequence):
        return [str(arg) for arg in run_cfg], None, f"run-{idx}"
    raise ValueError("Each entry in batch.runs must be a mapping or sequence")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
    return value


def _to_plain_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _to_plain_dict(value) for key, value in data.items()}
    return _normalize_value(data)


def _apply_cli_overrides(
    base_cfg: Mapping[str, Any],
    args_list: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    """
    Merge CLI-style overrides into the base configuration.

    Returns the merged configuration dictionary and any explicitly provided
    ``--suffix`` value (even if empty). The suffix is returned separately so the
    caller can compose it with existing suffix entries if desired.
    """
    applied_cfg = deepcopy(dict(base_cfg))
    if not args_list:
        return applied_cfg, None

    overrides_raw = _parse_args_to_nested_dict(list(args_list))
    overrides = _to_plain_dict(overrides_raw)

    # Remove keys that are not relevant for config overrides.
    suffix_override = None
    if isinstance(overrides, dict) and "suffix" in overrides:
        suffix_override = overrides.pop("suffix")
        if suffix_override is not None:
            suffix_override = str(suffix_override)

    merge_nested_dicts(applied_cfg, overrides if isinstance(overrides, dict) else {})
    return applied_cfg, suffix_override


def _prepare_run_cleanup(
    config_path: str,
    extra_args: Sequence[Any] | None,
    *,
    dry_run: bool,
    label: str | None = None,
) -> list[str] | None:
    """
    Construct the `sweagent run-batch` command and emit dry-run/run logging.

    Returns the command to execute when `dry_run` is False; otherwise prints the
    simulated command and returns None.
    """
    cmd = [
        sys.executable,
        "-m",
        "sweagent.run.run",
        "run-batch",
        "--config",
        config_path,
    ]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)

    if dry_run:
        prefix = "[dry-run] "
        if label:
            prefix += f"[{label}] "
        print(prefix + " ".join(cmd))
        return None

    if label:
        print(f"[run] Starting batch '{label}'")
    return cmd


def run_batch(
    config_path: str,
    extra_args: Sequence[Any] | None,
    *,
    env: Mapping[str, Any] | None,
    dry_run: bool,
    label: str | None = None,
) -> int:
    cmd = _prepare_run_cleanup(config_path, extra_args, dry_run=dry_run, label=label)
    if dry_run or cmd is None:
        return 0
    completed = subprocess.run(cmd, env=_merge_env(env))
    return completed.returncode


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config_file)

    server_cfg = ensure_mapping(raw_config.get("vllm"), "vllm")
    batch_cfg = ensure_mapping(raw_config.get("batch"), "batch")

    sweagent_config = str(batch_cfg.get("sweagent_config", "config/vllm_local.yaml"))
    batch_args = batch_cfg.get("args", [])
    runs = batch_cfg.get("runs")
    allowed_runs: set[str] | None = None
    if batch_args and not isinstance(batch_args, Sequence):
        raise ValueError("batch.args must be a sequence if provided")
    if runs is not None and not isinstance(runs, Sequence):
        raise ValueError("batch.runs must be a sequence if provided")
    if args.only_runs:
        allowed_runs = {str(item) for item in args.only_runs}
        if not allowed_runs:
            allowed_runs = None

    server_cmd = build_vllm_command(server_cfg)
    print("Starting vLLM server:", " ".join(server_cmd))

    if args.dry_run:
        print("[dry-run] would wait for server and run sweagent batch")
        if runs:
            matched = False
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = _extract_run(run_cfg, idx)
                if allowed_runs and label not in allowed_runs and str(idx) not in allowed_runs and str(idx + 1) not in allowed_runs:
                    continue
                matched = True
                combined_args = list(batch_args) + run_args
                combined_env = _merge_two_envs(batch_cfg.get("env"), run_env)
                run_batch(sweagent_config, combined_args, env=combined_env, dry_run=True, label=label)
            if allowed_runs and not matched:
                print(f"[dry-run] No runs matched --only-runs {sorted(allowed_runs)}")
        else:
            run_batch(sweagent_config, batch_args, env=batch_cfg.get("env"), dry_run=True)
        return 0

    server_proc = subprocess.Popen(server_cmd, env=_merge_env(server_cfg.get("env")))
    try:
        connect_host = str(server_cfg.get("connect_host", server_cfg.get("host", "127.0.0.1")))
        port = int(server_cfg.get("port", 8000))
        timeout = int(server_cfg.get("wait_timeout", raw_config.get("wait_timeout", 180)))
        wait_for_server(connect_host, port, timeout)
        print("vLLM server is ready. Running sweagent batch...")
        exit_code = 0
        if runs:
            matched = False
            for idx, run_cfg in enumerate(runs):
                run_args, run_env, label = _extract_run(run_cfg, idx)
                if allowed_runs and label not in allowed_runs and str(idx) not in allowed_runs and str(idx + 1) not in allowed_runs:
                    continue
                matched = True
                combined_args = list(batch_args) + run_args
                combined_env = _merge_two_envs(batch_cfg.get("env"), run_env)
                exit_code = run_batch(
                    sweagent_config,
                    combined_args,
                    env=combined_env,
                    dry_run=False,
                    label=label,
                )
                if exit_code != 0:
                    print(f"Batch '{label}' failed with exit code {exit_code}; stopping remaining runs")
                    break
            if allowed_runs and not matched:
                print(f"No runs matched --only-runs {sorted(allowed_runs)}")
                exit_code = 0
        else:
            exit_code = run_batch(
                sweagent_config,
                batch_args,
                env=batch_cfg.get("env"),
                dry_run=False,
            )
        return exit_code
    finally:
        print("Stopping vLLM server...")
        terminate_process(server_proc)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())