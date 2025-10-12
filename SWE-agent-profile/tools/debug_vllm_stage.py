#!/usr/bin/env python3
"""Launch a single vLLM stage and surface startup issues.

Usage:

    python3 tools/debug_vllm_stage.py --stage prefill

The script reuses the settings in ``config/vllm_batch.yaml`` so you can
reproduce exactly what ``tools/run_vllm_batch.py`` would do, but for one
stage at a time. It prints the command/environment, tails the log when
startup fails, and exits with a non-zero status on errors.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_vllm_batch import (
    build_vllm_command,
    ensure_mapping,
    load_config,
    wait_for_server,
    _merge_env,
    _merge_two_envs,
    _normalize_server_configs,
    _resolve_server_value,
)


def _tail(path: Path, limit: int = 60) -> str:
    if not path.exists():
        return "<log file missing>"
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if limit > 0 and len(lines) > limit:
        lines = lines[-limit:]
    return "".join(lines) or "<log file empty>"


def _pretty_env(env: dict[str, Any] | None) -> str:
    if not env:
        return "<none>"
    return "\n".join(f"  {k}={v}" for k, v in sorted(env.items()))


def debug_stage(config_path: Path, stage: str, *, log_tail: int = 80, keep_alive: bool = False) -> int:
    cfg = load_config(config_path)
    vllm_cfg = ensure_mapping(cfg.get("vllm"), "vllm")
    stages = {label: entry for label, entry in _normalize_server_configs(vllm_cfg)}

    if stage not in stages:
        available = ", ".join(sorted(stages.keys())) or "<none>"
        raise SystemExit(f"Stage '{stage}' not found. Available: {available}")

    stage_cfg = stages[stage]
    merged_env = _merge_two_envs(vllm_cfg.get("env"), stage_cfg.get("env"))
    command = build_vllm_command(stage_cfg)

    host = str(
        _resolve_server_value(
            stage_cfg,
            vllm_cfg,
            "connect_host",
            _resolve_server_value(stage_cfg, vllm_cfg, "host", "127.0.0.1"),
        )
    )
    port = int(_resolve_server_value(stage_cfg, vllm_cfg, "port", 8000))
    timeout = int(_resolve_server_value(stage_cfg, vllm_cfg, "wait_timeout", cfg.get("wait_timeout", 180)))

    print(f"[debug] Config: {config_path}")
    print(f"[debug] Stage: {stage}")
    print(f"[debug] Command:\n  {' '.join(command)}")
    print("[debug] Env overrides:")
    print(_pretty_env(merged_env))
    print(f"[debug] Waiting for http://{host}:{port}/v1/models (timeout {timeout}s)\n")

    tmpdir = tempfile.mkdtemp(prefix=f"vllm-{stage}-debug-")
    log_path = Path(tmpdir) / "vllm.log"
    proc: subprocess.Popen[Any] | None = None

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                env=_merge_env(merged_env),
                stdout=log_file,
                stderr=log_file,
            )

        try:
            wait_for_server(host, port, timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[debug] Stage failed to become ready: {exc}")
            if proc and proc.poll() is not None:
                print(f"[debug] Process exited with code {proc.returncode}")
            print("[debug] Last log lines:\n" + "-" * 60)
            print(_tail(log_path, log_tail))
            print("-" * 60)
            return 1

        print("[debug] Stage reported ready.")
        if keep_alive:
            print(f"[debug] Process left running. Logs at: {log_path}")
            proc.wait()
        return 0
    finally:
        if proc and (proc.poll() is None) and not keep_alive:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not keep_alive:
            log_path.unlink(missing_ok=True)
            Path(tmpdir).rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a single vLLM stage defined in vllm_batch.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/vllm_batch.yaml"),
        help="Path to the batch configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--stage",
        default="prefill",
        help="Stage name to launch (e.g., prefill, decode)",
    )
    parser.add_argument(
        "--log-tail",
        type=int,
        default=80,
        help="Number of log lines to print when startup fails (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the process running after readiness (logs are kept).",
    )

    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    try:
        return debug_stage(args.config, args.stage, log_tail=args.log_tail, keep_alive=args.keep_alive)
    except KeyboardInterrupt:
        print("\n[debug] Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
