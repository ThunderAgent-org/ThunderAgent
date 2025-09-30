#!/usr/bin/env python3
"""Sequential SWE-bench runner with retry, cleanup, and evaluation.

Example:
    python tools/run_lite_sequential.py \
        --config config/default.yaml \
        --subset lite \
        --split test \
        --output-root trajectories/sequential_lite_test \
        --evaluate --run-id nightly_eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml
from swerex.exceptions import DockerPullError
from sweagent.run.batch_instances import SWEBenchInstances, BatchInstance
from sweagent.run.run_single import RunSingle, RunSingleConfig, RunSingleActionConfig
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.hooks.open_pr import OpenPRHook
from sweagent.run.run_batch import RunBatchConfig
from sweagent.utils.profiling import ProfilingAggregator

DATASET_NAME = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "multimodal": "princeton-nlp/SWE-bench_Multimodal",
}


def docker_cleanup(base_image: str) -> None:
    subprocess.run(["docker", "container", "prune", "-f"], check=False)
    if base_image:
        subprocess.run(["docker", "image", "rm", "-f", base_image], check=False)
    subprocess.run(["docker", "image", "prune", "-f"], check=False)


def build_run_single(batch_cfg: RunBatchConfig, instance: BatchInstance, output_root: Path) -> RunSingle:
    actions = getattr(batch_cfg, "actions", None)
    if actions is None:
        actions = RunSingleActionConfig()

    single_cfg = RunSingleConfig(
        agent=batch_cfg.agent,
        env=instance.env,
        problem_statement=instance.problem_statement,
        actions=actions,
        output_dir=output_root,
        env_var_path=batch_cfg.env_var_path,
    )
    run_single = RunSingle.from_config(single_cfg)
    run_single.remove_hook = []
    run_single.add_hook(SaveApplyPatchHook(apply_patch_locally=actions.apply_patch_locally))
    if actions.open_pr:
        run_single.add_hook(OpenPRHook(actions.pr_config))
    return run_single


def run_instance(
    batch_cfg: RunBatchConfig,
    instance: BatchInstance,
    output_root: Path,
    retries: int,
    sleep_between: float,
) -> bool:
    for attempt in range(1, retries + 1):
        run_single = build_run_single(batch_cfg, instance, output_root)
        base_image = instance.env.deployment.image if instance.env.deployment else ""
        try:
            run_single.run()
            return True
        except DockerPullError as exc:
            print(f"[WARN] Docker pull failed ({attempt}/{retries}): {exc}")
        except subprocess.CalledProcessError as exc:
            if "docker" in str(exc.cmd):
                print(f"[WARN] Docker command failed ({attempt}/{retries}): {exc}")
            else:
                raise
        except RuntimeError as exc:
            if "ports are not available" in str(exc):
                print(f"[WARN] Port allocation failed ({attempt}/{retries}): {exc}")
            else:
                raise
        finally:
            docker_cleanup(base_image)
        time.sleep(sleep_between)
    print("[FAIL] Max retries exceeded for this instance.")
    return False


def collect_success_ids(output_root: Path) -> set[str]:
    success = set()
    for profiling_path in output_root.glob("*/profiling.json"):
        try:
            data = json.loads(profiling_path.read_text())
        except Exception:
            continue
        if data.get("status") == "success":
            success.add(profiling_path.parent.name)
    return success


def write_success_predictions(output_root: Path, success_ids: set[str]) -> Path:
    preds_path = output_root / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found")

    content = preds_path.read_text().strip()
    if not content:
        raise RuntimeError("preds.json is empty")

    filtered: list[dict[str, object]] = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = data.values()
        else:
            iterable = [data]
        for obj in iterable:
            if obj.get("instance_id") in success_ids:
                filtered.append(obj)
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if obj.get("instance_id") in success_ids:
                filtered.append(obj)

    out_path = output_root / "preds_success.jsonl"
    out_path.write_text("\n".join(json.dumps(obj) for obj in filtered))
    print(f"Wrote {len(filtered)} success predictions -> {out_path}")
    return out_path


def parse_missing_images(log_text: str) -> list[str]:
    items = []
    for line in log_text.splitlines():
        if "No such image" in line or "failed to resolve reference" in line:
            items.append(line.strip())
    return items


def apply_evaluation(batch_dir: Path, results_path: Path) -> None:
    if not results_path.exists():
        print(f"[WARN] Evaluation results not found at {results_path}")
        return

    results = json.loads(results_path.read_text())
    resolved = set(results.get("resolved_ids", []))
    unresolved = set(results.get("unresolved_ids", []))
    error_ids = set(results.get("error_ids", []))

    aggregator = ProfilingAggregator()

    for profiling_path in batch_dir.glob("*/profiling.json"):
        data = json.loads(profiling_path.read_text())
        inst_id = data.get("problem_id") or profiling_path.parent.name
        if inst_id in resolved:
            status = "success"
        else:
            status = "failure"
        data["status"] = status
        metadata = data.setdefault("metadata", {})
        metadata["evaluation_status"] = status
        metadata["evaluation_source"] = str(results_path)
        if inst_id in error_ids:
            metadata["evaluation_note"] = "missing_image"
        profiling_path.write_text(json.dumps(data, indent=2))
        aggregator.add(data)

    summary = aggregator.summary()
    if summary is None:
        print("[WARN] No profiling data to summarise after evaluation.")
        return

    summary.setdefault("evaluation", {})
    summary["evaluation"].update(
        {
            "resolved_ids": sorted(resolved),
            "unresolved_ids": sorted(unresolved),
            "error_ids": sorted(error_ids),
            "results_json": str(results_path),
        }
    )

    summary_path = batch_dir / "profiling_summary.json"
    if summary_path.exists():
        try:
            old = json.loads(summary_path.read_text())
            if "configuration" in old:
                summary["configuration"] = old["configuration"]
        except Exception:
            pass
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Updated profiling_summary.json (success rate {summary['success_rate']['percentage']*100:.2f}%)."
    )


def build_summary_without_eval(batch_dir: Path) -> None:
    aggregator = ProfilingAggregator()
    for profiling_path in batch_dir.glob("*/profiling.json"):
        try:
            data = json.loads(profiling_path.read_text())
        except Exception:
            continue
        aggregator.add(data)
    summary = aggregator.summary()
    if summary is None:
        print("[WARN] No profiling data to summarise.")
        return
    summary_path = batch_dir / "profiling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Wrote profiling_summary.json (success rate {summary['success_rate']['percentage']*100:.2f}%)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential SWE-bench runner with evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", default="lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-between", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=Path("trajectories/sequential"))
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run-id", default="sequential_eval")
    parser.add_argument("--eval-max-workers", type=int, default=8)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg_data = yaml.safe_load(cfg_path.read_text()) or {}
    cfg_data.setdefault(
        "instances",
        {"type": "swe_bench", "subset": args.subset, "split": args.split},
    )
    batch_cfg = RunBatchConfig.model_validate(cfg_data)
    batch_cfg.instances = SWEBenchInstances(subset=args.subset, split=args.split)
    batch_cfg.num_workers = 1

    instances = batch_cfg.instances.get_instance_configs()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(instances)} instances from subset={args.subset}, split={args.split}")
    failed: list[str] = []

    for idx, instance in enumerate(instances, 1):
        inst_id = instance.problem_statement.id
        inst_dir = output_root / inst_id
        if (inst_dir / "profiling.json").exists():
            print(f"[SKIP] {inst_id} already processed at {inst_dir}.")
            continue

        print(f"\n[{idx}/{len(instances)}] Running {inst_id}")
        success = run_instance(
            batch_cfg,
            instance,
            output_root,
            args.max_retries,
            args.sleep_between,
        )
        if not success:
            failed.append(inst_id)
            print(f"[SKIP] {inst_id} failed after retries.")

    print("\nSequential run finished.")
    if failed:
        print(f"Failed instances: {', '.join(failed)}")

    success_ids = collect_success_ids(output_root)
    print(f"Collected {len(success_ids)} success directories.")
    if not success_ids:
        build_summary_without_eval(output_root)
        return

    preds_success = write_success_predictions(output_root, success_ids)

    if not args.evaluate:
        build_summary_without_eval(output_root)
        return

    dataset_name = DATASET_NAME.get(args.subset, f"princeton-nlp/SWE-bench_{args.subset.capitalize()}")
    results_path = preds_success.with_name(f"{preds_success.name}.{args.run_id}.json")
    missing_file = output_root / "evaluation_missing_images.txt"

    eval_cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        str(preds_success),
        "--max_workers",
        str(args.eval_max_workers),
        "--run_id",
        args.run_id,
    ]
    print("\nRunning evaluation:", " ".join(eval_cmd))
    proc = subprocess.run(eval_cmd, capture_output=True, text=True)
    eval_log = output_root / f"evaluation_{args.run_id}.log"
    eval_log.write_text(proc.stdout + "\n" + proc.stderr)

    if proc.returncode != 0:
        print(f"[WARN] Evaluation exited with code {proc.returncode}. See {eval_log}")
        missing = parse_missing_images(proc.stdout + proc.stderr)
        if missing:
            missing_file.write_text("\n".join(missing) + "\n")
            print(f"Recorded missing images to {missing_file}")
        else:
            print("No missing-image lines detected in evaluation output.")

    if results_path.exists():
        apply_evaluation(output_root, results_path)
    else:
        print(f"[WARN] Results file not found at {results_path}; writing summary without evaluation.")
        build_summary_without_eval(output_root)


if __name__ == "__main__":
    main()
