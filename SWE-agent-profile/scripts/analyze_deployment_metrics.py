from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_json_lines(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            yield json.loads(line)


def format_hours_minutes(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} 小时 {minutes} 分钟"


def analyze_strategy(strategy_dir: Path) -> None:
    stage_files = sorted(strategy_dir.glob("**/stage_timings.jsonl"))
    start_times: list[float] = []
    total_prefill = 0.0
    total_decode = 0.0
    total_steps = 0
    instances = 0
    latest_timestamp: float | None = None
    latest_hits_after: float | None = None
    latest_queries_after: float | None = None

    for stage_path in stage_files:
        prefix_path = stage_path.with_name("prefix_cache_metrics.jsonl")
        if not prefix_path.exists():
            continue

        instances += 1

        stage_timestamps = [record["timestamp"] for record in iter_json_lines(stage_path)]
        if stage_timestamps:
            start_times.append(min(stage_timestamps))

        for record in iter_json_lines(prefix_path):
            prefill = float(record.get("prefill_time", 0.0))
            decode = float(record.get("decode_time", 0.0))
            total_prefill += prefill
            total_decode += decode
            total_steps += 1

            timestamp = float(record.get("timestamp", 0.0))
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_hits_after = float(record.get("prefix_cache_hits_after", 0.0))
                latest_queries_after = float(
                    record.get("prefix_cache_queries_after", 0.0)
                )

    print(f"部署策略: {strategy_dir.name}")
    print(f"实例数量: {instances}")

    if not start_times:
        print("未找到任何起始时间\n")
        return

    earliest = min(start_times)
    latest = max(start_times)
    duration = latest - earliest
    print(f"起始时间范围: {earliest:.3f} → {latest:.3f}")
    formatted_duration = format_hours_minutes(duration)
    print(f"总时长: {formatted_duration} (≈ {duration:.3f} 秒)")

    if total_steps == 0:
        print("未找到任何 step 数据\n")
        return

    avg_prefill = total_prefill / total_steps
    avg_decode = total_decode / total_steps
    print(f"平均 prefill 时间: {avg_prefill:.6f} 秒 (基于 {total_steps} 个 step)")
    print(f"平均 decode 时间: {avg_decode:.6f} 秒 (基于 {total_steps} 个 step)")

    if (
        latest_timestamp is not None
        and latest_hits_after is not None
        and latest_queries_after is not None
        and latest_queries_after > 0.0
    ):
        hit_rate = latest_hits_after / latest_queries_after
        print(
            "最新时间戳: "
            f"{latest_timestamp:.3f}, 命中率: {hit_rate:.6f}"
            f" (hits_after={latest_hits_after:.0f}, queries_after={latest_queries_after:.0f})\n"
        )
    else:
        print("未找到用于计算命中率的有效数据\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析不同部署策略的前缀缓存指标")
    parser.add_argument(
        "--testdata-root",
        type=Path,
        default=Path("/home/ziyang/multiagent/upload/SWE-agent-profile/testdata"),
        help="testdata 根目录",
    )
    args = parser.parse_args()

    testdata_root = args.testdata_root.resolve()
    if not testdata_root.exists():
        raise FileNotFoundError(f"未找到目录: {testdata_root}")

    strategy_dirs = sorted(
        directory for directory in testdata_root.iterdir() if directory.is_dir()
    )

    if not strategy_dirs:
        raise ValueError(f"目录 {testdata_root} 下未发现部署策略子目录")

    for strategy_dir in strategy_dirs:
        analyze_strategy(strategy_dir)


if __name__ == "__main__":
    main()

