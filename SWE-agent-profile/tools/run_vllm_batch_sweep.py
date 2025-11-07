#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Variant:
    name: str
    tensor_parallel: int | None
    pipeline_parallel: int | None
    data_parallel: int
    enable_expert_parallel: bool


VARIANTS: list[Variant] = [
    Variant("tp8_ep", tensor_parallel=8, pipeline_parallel=None, enable_expert_parallel=True),
]

VARIANT_MAP: dict[str, Variant] = {variant.name: variant for variant in VARIANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep vLLM batch runs across multiple parallelism configurations.")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/vllm_batch.yaml"),
        help="Path to the vLLM batch YAML config to mutate during the sweep.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root used as the working directory for subprocess calls.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("tools/run_vllm_batch_nonblocking.py"),
        help="Path to the non-blocking batch runner script.",
    )
    parser.add_argument(
        "--python-exec",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch the batch runner.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["bs64"],
        help="Run labels to forward to --only-runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of variant names to execute (default: all).",
    )
    parser.add_argument(
        "--trajectories-user",
        default="ziyang",
        help="Subdirectory under trajectories/ containing run outputs to archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("testdata"),
        help="Directory where archived trajectories will be stored.",
    )
    parser.add_argument(
        "--skip-command",
        action="store_true",
        help="Skip executing the batch runner (useful for testing config mutations).",
    )
    parser.add_argument(
        "--skip-docker-cleanup",
        action="store_true",
        help="Do not attempt to remove dangling docker images between runs.",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip moving trajectory outputs after each run.",
    )
    parser.add_argument(
        "--keep-modified-config",
        action="store_true",
        help="Do not restore the original config after the sweep (final variant remains on disk).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def dump_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def strip_flag_with_value(args: list[Any], flag: str) -> list[Any]:
    result: list[Any] = []
    i = 0
    while i < len(args):
        entry = args[i]
        if str(entry) == flag:
            i += 2
            continue
        result.append(entry)
        i += 1
    return result


def strip_flag(args: list[Any], flag: str) -> list[Any]:
    return [entry for entry in args if str(entry) != flag]


def apply_variant(base: dict[str, Any], variant: Variant) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    vllm_cfg = cfg.setdefault("vllm", {})
    vllm_cfg.pop("tensor_parallel", None)
    extra_raw = list(vllm_cfg.get("extra_args", []) or [])
    extra = strip_flag_with_value(extra_raw, "--tensor-parallel-size")
    extra = strip_flag_with_value(extra, "--data-parallel-size")
    extra = strip_flag_with_value(extra, "--pipeline-parallel-size")
    extra = strip_flag(extra, "--enable-expert-parallel")
    if variant.enable_expert_parallel:
        extra.append("--enable-expert-parallel")
    if variant.tensor_parallel is not None:
        extra.extend(["--tensor-parallel-size", variant.tensor_parallel])
    extra.extend(["--data-parallel-size", variant.data_parallel])
    if variant.pipeline_parallel is not None:
        extra.extend(["--pipeline-parallel-size", variant.pipeline_parallel])
    vllm_cfg["extra_args"] = extra
    return cfg


def run_command(python_exec: Path, runner: Path, selected_runs: list[str], config_path: Path, workdir: Path) -> None:
    cmd = [
        str(python_exec),
        str(runner),
        "--config-file",
        str(config_path),
        "--only-runs",
        *selected_runs,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workdir, check=True)


def cleanup_dangling_images() -> None:
    probe = subprocess.run(
        ["docker", "images", "--filter", "dangling=true", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return
    ids = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not ids:
        return
    subprocess.run(["docker", "rmi", *ids], check=False)


def archive_trajectories(repo_root: Path, user: str, output_root: Path, variant: Variant) -> None:
    source = repo_root / "trajectories" / user
    if not source.exists():
        return
    items = list(source.iterdir())
    if not items:
        return
    target = output_root / variant.name
    target.mkdir(parents=True, exist_ok=True)
    for item in items:
        destination = target / item.name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        item.rename(destination)


def main() -> int:
    args = parse_args()
    config_path = args.config_file.resolve()
    repo_root = args.repo_root.resolve()
    runner_path = args.runner if args.runner.is_absolute() else (repo_root / args.runner)
    runner_path = runner_path.resolve()
    python_exec = args.python_exec.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else (repo_root / args.output_root)
    output_root = output_root.resolve()

    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    original_text = config_path.read_text()
    base_config = load_config(config_path)

    if args.variants:
        selected_variants: list[Variant] = []
        for name in args.variants:
            matched = VARIANT_MAP.get(name)
            if matched is None:
                raise ValueError(f"Unknown variant name '{name}'. Available: {sorted(VARIANT_MAP)}")
            selected_variants.append(matched)
    else:
        selected_variants = VARIANTS

    try:
        for variant in selected_variants:
            print(f"=== Variant {variant.name} ===")
            mutated = apply_variant(base_config, variant)
            dump_config(config_path, mutated)
            if not args.skip_command:
                try:
                    run_command(python_exec, runner_path, list(args.runs), config_path, repo_root)
                except subprocess.CalledProcessError as exc:
                    print(f"Variant {variant.name} failed with return code {exc.returncode}", file=sys.stderr)
                    raise
            if not args.skip_docker_cleanup:
                cleanup_dangling_images()
            if not args.skip_archive:
                archive_trajectories(repo_root, args.trajectories_user, output_root, variant)
    finally:
        if args.keep_modified_config:
            print("[info] keep-modified-config enabled; retaining last variant in config file")
        else:
            config_path.write_text(original_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

