#!/usr/bin/env python3
"""
分析 timing 文件夹中所有 JSON 文件的 tool 执行时间，生成密度分布图。
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

TIMING_DIR = Path("/mnt/shared/MultiagentSystem/openhands_swebench/OpenHands/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/GLM-4.6-FP8_maxiter_50/timing")


def load_tool_times(timing_dir: Path) -> dict:
    """
    从所有 timing 文件中提取 tool_time 数据。
    
    Returns:
        {
            "all": [所有 tool_time 列表],
            "by_instance": {instance_id: [tool_times]},
            "env_prepare": [环境准备时间列表],
        }
    """
    all_tool_times = []
    by_instance = defaultdict(list)
    env_prepare_times = []
    
    for json_file in sorted(timing_dir.glob("timing_*.json")):
        with open(json_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                instance_id = record.get("instance_id", "unknown")
                
                # 提取 tool_time
                if "tool_time" in record:
                    tool_time = record["tool_time"]
                    all_tool_times.append(tool_time)
                    by_instance[instance_id].append(tool_time)
                
                # 提取环境准备时间
                if "env_prepare_time" in record:
                    env_prepare_times.append(record["env_prepare_time"])
    
    return {
        "all": all_tool_times,
        "by_instance": dict(by_instance),
        "env_prepare": env_prepare_times,
    }


def plot_density(data: list, title: str, xlabel: str, output_path: Path, 
                 bins: int = 100, log_scale: bool = False, xlim: tuple = None):
    """绘制直方图，纵坐标为每个区间内的调用次数"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = np.array(data)
    
    # 计算统计信息
    stats = {
        "count": len(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "p50": np.percentile(data, 50),
        "p90": np.percentile(data, 90),
        "p95": np.percentile(data, 95),
        "p99": np.percentile(data, 99),
    }
    
    # 绘制直方图
    if log_scale and np.min(data) > 0:
        # 对数刻度
        log_data = np.log10(data)
        counts, bin_edges = np.histogram(log_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=bin_width * 0.9, 
               alpha=0.7, color='steelblue', edgecolor='white')
        ax.set_xlabel(f"{xlabel} (log10 scale)")
        
        # 添加原始刻度标签
        tick_vals = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        tick_vals = tick_vals[(tick_vals >= data.min() * 0.5) & (tick_vals <= data.max() * 2)]
        ax.set_xticks(np.log10(tick_vals))
        ax.set_xticklabels([f"{v:.3g}" for v in tick_vals])
    else:
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=bin_width * 0.9, 
               alpha=0.7, color='steelblue', edgecolor='white')
        ax.set_xlabel(xlabel)
    
    if xlim:
        ax.set_xlim(xlim)
    
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    # 添加统计信息文本框
    stats_text = (
        f"Count: {stats['count']:,}\n"
        f"Mean: {stats['mean']:.3f}s\n"
        f"Median: {stats['median']:.3f}s\n"
        f"Std: {stats['std']:.3f}s\n"
        f"Min: {stats['min']:.3f}s\n"
        f"Max: {stats['max']:.3f}s\n"
        f"P90: {stats['p90']:.3f}s\n"
        f"P95: {stats['p95']:.3f}s\n"
        f"P99: {stats['p99']:.3f}s"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    # 添加中位数和均值线
    if log_scale and np.min(data) > 0:
        ax.axvline(np.log10(stats['median']), color='red', linestyle='--', 
                   linewidth=2, label=f"Median: {stats['median']:.3f}s")
        ax.axvline(np.log10(stats['mean']), color='orange', linestyle='--', 
                   linewidth=2, label=f"Mean: {stats['mean']:.3f}s")
    else:
        ax.axvline(stats['median'], color='red', linestyle='--', 
                   linewidth=2, label=f"Median: {stats['median']:.3f}s")
        ax.axvline(stats['mean'], color='orange', linestyle='--', 
                   linewidth=2, label=f"Mean: {stats['mean']:.3f}s")
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return stats


def plot_time_ranges(data: list, title: str, output_path: Path):
    """绘制不同时间范围的分布饼图"""
    data = np.array(data)
    
    ranges = [
        ("< 0.01s", (0, 0.01)),
        ("0.01-0.1s", (0.01, 0.1)),
        ("0.1-0.5s", (0.1, 0.5)),
        ("0.5-1s", (0.5, 1.0)),
        ("1-5s", (1.0, 5.0)),
        ("5-10s", (5.0, 10.0)),
        ("10-60s", (10.0, 60.0)),
        ("> 60s", (60.0, float('inf'))),
    ]
    
    counts = []
    labels = []
    for label, (low, high) in ranges:
        count = np.sum((data >= low) & (data < high))
        if count > 0:
            counts.append(count)
            labels.append(f"{label}\n({count}, {count/len(data)*100:.1f}%)")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='',
                                       colors=colors, startangle=90)
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    print(f"Loading timing data from: {TIMING_DIR}")
    
    # 加载数据
    data = load_tool_times(TIMING_DIR)
    
    all_tool_times = data["all"]
    env_prepare_times = data["env_prepare"]
    
    print(f"\n=== Tool Execution Time Statistics ===")
    print(f"Total tool calls: {len(all_tool_times):,}")
    print(f"Total instances: {len(data['by_instance']):,}")
    print(f"Total env_prepare records: {len(env_prepare_times):,}")
    
    if not all_tool_times:
        print("No tool_time data found!")
        return
    
    output_dir = Path("/mnt/shared/MultiagentSystem/openhands_swebench")
    
    # 1. 全部 tool_time 的密度分布（线性）
    stats = plot_density(
        all_tool_times,
        "Tool Execution Time Distribution (All Tools)",
        "Time (seconds)",
        output_dir / "tool_time_density.png",
        bins=100,
        log_scale=False,
        xlim=(0, min(5, np.percentile(all_tool_times, 99) * 1.5))
    )
    
    # 2. 全部 tool_time 的密度分布（对数）
    plot_density(
        all_tool_times,
        "Tool Execution Time Distribution (Log Scale)",
        "Time (seconds)",
        output_dir / "tool_time_density_log.png",
        bins=80,
        log_scale=True
    )
    
    # 3. 时间范围分布饼图
    plot_time_ranges(
        all_tool_times,
        "Tool Execution Time Range Distribution",
        output_dir / "tool_time_pie.png"
    )
    
    # 4. 环境准备时间分布
    if env_prepare_times:
        plot_density(
            env_prepare_times,
            "Environment Preparation Time Distribution",
            "Time (seconds)",
            output_dir / "env_prepare_time_density.png",
            bins=50,
            log_scale=True
        )
    
    # 打印详细统计
    print(f"\n=== Detailed Statistics ===")
    print(f"Mean:   {stats['mean']:.4f}s")
    print(f"Median: {stats['median']:.4f}s")
    print(f"Std:    {stats['std']:.4f}s")
    print(f"Min:    {stats['min']:.4f}s")
    print(f"Max:    {stats['max']:.4f}s")
    print(f"P50:    {stats['p50']:.4f}s")
    print(f"P90:    {stats['p90']:.4f}s")
    print(f"P95:    {stats['p95']:.4f}s")
    print(f"P99:    {stats['p99']:.4f}s")
    
    # 分析快慢工具调用
    fast_calls = sum(1 for t in all_tool_times if t < 0.1)
    medium_calls = sum(1 for t in all_tool_times if 0.1 <= t < 1.0)
    slow_calls = sum(1 for t in all_tool_times if t >= 1.0)
    
    print(f"\n=== Call Speed Breakdown ===")
    print(f"Fast (< 0.1s):   {fast_calls:,} ({fast_calls/len(all_tool_times)*100:.1f}%)")
    print(f"Medium (0.1-1s): {medium_calls:,} ({medium_calls/len(all_tool_times)*100:.1f}%)")
    print(f"Slow (>= 1s):    {slow_calls:,} ({slow_calls/len(all_tool_times)*100:.1f}%)")


if __name__ == "__main__":
    main()

