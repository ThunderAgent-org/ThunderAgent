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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


def _normalize_env(raw_env: Mapping[str, Any] | None, *, context: str) -> dict[str, str]:
    if raw_env is None:
        return {}
    if not isinstance(raw_env, Mapping):
        raise ValueError(f"{context} must be a mapping of environment variables")
    return {str(key): str(value) for key, value in raw_env.items()}


def _resolve_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # noqa: BLE001
            return "127.0.0.1"


def _wait_for_tcp_port(host: str, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Server at {host}:{port} did not become ready within {timeout}s: {last_error}")


def _resolve_connect_host(cfg: Mapping[str, Any]) -> str:
    candidate = str(cfg.get("connect_host", cfg.get("host", "127.0.0.1")))
    if candidate in {"", "0.0.0.0"}:
        return "127.0.0.1"
    return candidate


@dataclass
class _ServerComponent:
    name: str
    command: list[str]
    env: Mapping[str, Any] | None
    connect_host: str
    port: int
    wait_timeout: int
    wait_mode: str = "openai"
    process: subprocess.Popen[Any] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError(f"{self.name} is already running")
        self.process = subprocess.Popen(self.command, env=blocking._merge_env(self.env))

    def wait_ready(self) -> None:
        if self.wait_mode == "openai":
            blocking.wait_for_server(self.connect_host, self.port, self.wait_timeout)
        else:
            _wait_for_tcp_port(self.connect_host, self.port, self.wait_timeout)

    def stop(self) -> None:
        if self.process is None:
            return
        blocking.terminate_process(self.process)
        self.process = None


class _ServerManagerBase:
    ready_message: str = ""
    stop_message: str = ""

    def print_startup_commands(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def wait_ready(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class _SingleServerManager(_ServerManagerBase):
    def __init__(self, server_cfg: Mapping[str, Any], default_timeout: int) -> None:
        self._server_cfg = server_cfg
        self._command = blocking.build_vllm_command(server_cfg)
        env_override = _normalize_env(server_cfg.get("env"), context="vllm.env")
        connect_host = _resolve_connect_host(server_cfg)
        port = int(server_cfg.get("port", 8000))
        wait_timeout = int(server_cfg.get("wait_timeout", default_timeout))
        self._component = _ServerComponent(
            name="vLLM server",
            command=self._command,
            env=env_override,
            connect_host=connect_host,
            port=port,
            wait_timeout=wait_timeout,
        )
        self.ready_message = "vLLM server is ready. Running sweagent batch (non-blocking)..."
        self.stop_message = "Stopping vLLM server..."

    def print_startup_commands(self) -> None:
        print("Starting vLLM server:", " ".join(self._command))

    def start(self) -> None:
        self._component.start()

    def wait_ready(self) -> None:
        self._component.wait_ready()

    def stop(self) -> None:
        self._component.stop()


class _DisaggregatedServerManager(_ServerManagerBase):
    def __init__(self, server_cfg: Mapping[str, Any], disagg_cfg: Mapping[str, Any], default_timeout: int) -> None:
        self._server_cfg = server_cfg
        self._disagg_cfg = disagg_cfg
        self.ready_message = "Disaggregated prefill proxy is ready. Running sweagent batch (non-blocking)..."
        self.stop_message = "Stopping disaggregated vLLM servers..."
        self._host_ip_value: str | None = None
        if bool(disagg_cfg.get("set_host_ip_env", True)):
            self._host_ip_value = _resolve_local_ip()
        self._shared_env = _normalize_env(disagg_cfg.get("env"), context="vllm.disaggregated_prefill.env")
        self._role_defaults = {
            key: server_cfg[key]
            for key in ("model", "tokenizer")
            if key in server_cfg
        }
        base_extra_args_raw = server_cfg.get("extra_args", []) or []
        if isinstance(base_extra_args_raw, Sequence) and not isinstance(base_extra_args_raw, str):
            self._base_extra_args = [str(arg) for arg in base_extra_args_raw]
        elif base_extra_args_raw:
            self._base_extra_args = [str(base_extra_args_raw)]
        else:
            self._base_extra_args = []
        prefill_cfg = blocking.ensure_mapping(disagg_cfg.get("prefill"), "vllm.disaggregated_prefill.prefill")
        raw_decode_cfg = disagg_cfg.get("decode")
        if raw_decode_cfg is None:
            raise ValueError("vllm.disaggregated_prefill.decode must be provided")
        if isinstance(raw_decode_cfg, Sequence) and not isinstance(raw_decode_cfg, Mapping):
            decode_cfgs = [blocking.ensure_mapping(item, "vllm.disaggregated_prefill.decode[]") for item in raw_decode_cfg]
        else:
            decode_cfgs = [blocking.ensure_mapping(raw_decode_cfg, "vllm.disaggregated_prefill.decode")]
        self._prefill_component, self._prefill_url = self._build_role_component(
            "prefill",
            prefill_cfg,
            default_timeout,
        )
        self._decode_components: list[_ServerComponent] = []
        self._decode_urls: list[str] = []
        for idx, cfg in enumerate(decode_cfgs):
            component, url = self._build_role_component(
                "decode",
                cfg,
                default_timeout,
                index=idx,
            )
            self._decode_components.append(component)
            self._decode_urls.append(url)
        self._proxy_component = self._build_proxy_component(default_timeout)

    def print_startup_commands(self) -> None:
        print("Starting vLLM prefill server:", " ".join(self._prefill_component.command))
        if len(self._decode_components) == 1:
            print("Starting vLLM decode server:", " ".join(self._decode_components[0].command))
        else:
            for comp in self._decode_components:
                print("Starting vLLM decode server:", " ".join(comp.command))
        print("Starting vLLM proxy server:", " ".join(self._proxy_component.command))

    def start(self) -> None:
        self._prefill_component.start()
        for component in self._decode_components:
            component.start()

    def wait_ready(self) -> None:
        self._prefill_component.wait_ready()
        for component in self._decode_components:
            component.wait_ready()
        if self._proxy_component.process is None:
            self._proxy_component.start()
        self._proxy_component.wait_ready()

    def stop(self) -> None:
        self._proxy_component.stop()
        for component in self._decode_components:
            component.stop()
        self._prefill_component.stop()

    def _build_role_component(
        self,
        role: str,
        role_cfg: Mapping[str, Any],
        default_timeout: int,
        index: int | None = None,
    ) -> tuple[_ServerComponent, str]:
        cfg = dict(role_cfg)
        for key, value in self._role_defaults.items():
            cfg.setdefault(key, value)
        connect_host = _resolve_connect_host(cfg)
        port_default = 8100 if role == "prefill" else 8200
        port = int(cfg.get("port", port_default))
        wait_timeout = int(cfg.pop("wait_timeout", self._disagg_cfg.get("wait_timeout", default_timeout)))
        if "port" not in cfg:
            cfg["port"] = port
        env_override = dict(self._shared_env)
        env_override.update(
            _normalize_env(cfg.pop("env", None), context=f"vllm.disaggregated_prefill.{role}.env")
        )
        cuda_devices = cfg.pop("cuda_visible_devices", cfg.pop("cuda_devices", None))
        if cuda_devices is not None and "CUDA_VISIBLE_DEVICES" not in env_override:
            env_override["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        if self._host_ip_value and "VLLM_HOST_IP" not in env_override:
            env_override["VLLM_HOST_IP"] = self._host_ip_value
        kv_cfg_raw = cfg.pop("kv_transfer_config", None)
        extra_args = list(self._base_extra_args)
        role_extra_args_raw = cfg.pop("extra_args", None)
        if role_extra_args_raw is not None:
            if isinstance(role_extra_args_raw, Sequence) and not isinstance(role_extra_args_raw, str):
                extra_args.extend(str(arg) for arg in role_extra_args_raw)
            else:
                extra_args.append(str(role_extra_args_raw))
        if kv_cfg_raw is not None:
            if not isinstance(kv_cfg_raw, Mapping):
                raise ValueError(f"vllm.disaggregated_prefill.{role}.kv_transfer_config must be a mapping")
            kv_cfg = dict(kv_cfg_raw)
            extra_args.extend(["--kv-transfer-config", json.dumps(kv_cfg, separators=(",", ":"))])
        cfg.pop("connect_host", None)
        service_url = role_cfg.get("service_url")
        service_scheme = str(role_cfg.get("service_scheme", "http"))
        service_path = str(role_cfg.get("service_path", "/v1/completions"))
        if not service_path.startswith("/"):
            service_path = f"/{service_path}"
        if extra_args:
            cfg["extra_args"] = extra_args
        cfg.pop("service_url", None)
        cfg.pop("service_scheme", None)
        cfg.pop("service_path", None)
        name_suffix = f"#{index}" if index is not None else ""
        component = _ServerComponent(
            name=f"{role} vLLM server{name_suffix}",
            command=blocking.build_vllm_command(cfg),
            env=env_override,
            connect_host=connect_host,
            port=port,
            wait_timeout=wait_timeout,
        )
        url = service_url or f"{service_scheme}://{connect_host}:{port}{service_path}"
        return component, url

    def _build_proxy_component(self, default_timeout: int) -> _ServerComponent:
        proxy_cfg_raw = self._disagg_cfg.get("proxy") or {}
        if not isinstance(proxy_cfg_raw, Mapping):
            raise ValueError("vllm.disaggregated_prefill.proxy must be a mapping")
        proxy_cfg = dict(proxy_cfg_raw)
        script_path = proxy_cfg.pop("script", REPO_ROOT / "example_PD_disagg/disagg_prefill_proxy_server.py")
        script_path = Path(script_path)
        if not script_path.is_absolute():
            script_path = (REPO_ROOT / script_path).resolve()
        connect_host_default = _resolve_connect_host(self._server_cfg)
        port = int(proxy_cfg.pop("port", self._server_cfg.get("port", 8000)))
        wait_timeout = int(proxy_cfg.pop("wait_timeout", self._disagg_cfg.get("wait_timeout", default_timeout)))
        connect_host = str(proxy_cfg.pop("connect_host", connect_host_default))
        prefill_url = str(proxy_cfg.pop("prefill_url", self._prefill_url))
        default_decode = self._decode_urls[0] if self._decode_urls else self._prefill_url
        decode_url = str(proxy_cfg.pop("decode_url", default_decode))
        env_override = _normalize_env(proxy_cfg.pop("env", None), context="vllm.disaggregated_prefill.proxy.env")
        extra_args = [str(arg) for arg in proxy_cfg.pop("extra_args", []) or []]
        if proxy_cfg:
            unexpected = ", ".join(sorted(str(key) for key in proxy_cfg))
            raise ValueError(f"Unsupported proxy configuration options: {unexpected}")
        command = [
            sys.executable,
            str(script_path),
            "--port",
            str(port),
            "--prefill-url",
            prefill_url,
            "--decode-url",
            decode_url,
        ]
        command.extend(extra_args)
        return _ServerComponent(
            name="disaggregated proxy",
            command=command,
            env=env_override,
            connect_host=connect_host,
            port=port,
            wait_timeout=wait_timeout,
            wait_mode="tcp",
        )


def _create_server_manager(server_cfg: Mapping[str, Any], raw_config: Mapping[str, Any]) -> _ServerManagerBase:
    default_timeout = int(server_cfg.get("wait_timeout", raw_config.get("wait_timeout", 180)))
    disagg_cfg = server_cfg.get("disaggregated_prefill")
    if disagg_cfg is not None:
        disagg_mapping = blocking.ensure_mapping(disagg_cfg, "vllm.disaggregated_prefill")
        return _DisaggregatedServerManager(server_cfg, disagg_mapping, default_timeout)
    return _SingleServerManager(server_cfg, default_timeout)


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


def _dry_run_batches(
    sweagent_config: str,
    batch_args: Sequence[str],
    batch_cfg: Mapping[str, Any],
    runs: Sequence[Any] | None,
    selected_runs: set[str],
) -> set[str]:
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
    return seen


def _execute_batches(
    sweagent_config: str,
    batch_args: Sequence[str],
    batch_cfg: Mapping[str, Any],
    runs: Sequence[Any] | None,
    selected_runs: set[str],
) -> tuple[int, set[str]]:
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
    return exit_code, seen


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
    server_manager = _create_server_manager(server_cfg, raw_config)
    server_manager.print_startup_commands()

    if args.dry_run:
        print("[dry-run] would wait for server and run sweagent batch (non-blocking)")
        seen = _dry_run_batches(sweagent_config, batch_args, batch_cfg, runs, selected_runs)
        if selected_runs and selected_runs - seen:
            missing = ", ".join(sorted(selected_runs - seen))
            raise ValueError(f"Requested runs not found in config: {missing}")
        return 0

    started = False
    try:
        server_manager.start()
        started = True
        server_manager.wait_ready()
        print(server_manager.ready_message)
        exit_code, seen = _execute_batches(sweagent_config, batch_args, batch_cfg, runs, selected_runs)
        if selected_runs and selected_runs - seen:
            missing = ", ".join(sorted(selected_runs - seen))
            raise ValueError(f"Requested runs not found in config: {missing}")
        return exit_code
    finally:
        if started:
            print(server_manager.stop_message)
        server_manager.stop()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
