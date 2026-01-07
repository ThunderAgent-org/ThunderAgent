#!/usr/bin/env python3
"""
分析timing文件夹中的JSON文件
- 找到最开始的时间
- 找到10分钟后的时间点
- 统计接下来2小时内的LLM步骤数量
"""

import json
import os
from pathlib import Path
from collections import defaultdict

TIMING_DIR = "/mnt/shared/MultiagentSystem/openhands_swebench/OpenHands/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/GLM-4.6-FP8_maxiter_50/timing"

TEN_MINUTES = 10 * 60  # 600 秒
TWO_HOURS = 2 * 60 * 60  # 7200 秒


def parse_timing_file(filepath):
    """解析一个timing文件，返回所有记录"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    return records


def get_record_time(record):
    """获取记录的时间戳"""
    # 优先使用llm_start（如果是LLM记录）
    if 'llm_start' in record:
        return record['llm_start']
    # 然后是tool_start
    if 'tool_start' in record:
        return record['tool_start']
    # 环境准备记录
    if 'env_prepare_start' in record:
        return record['env_prepare_start']
    return None


def is_llm_record(record):
    """判断是否是LLM调用记录"""
    return 'llm_start' in record


def analyze_all_files():
    """分析所有timing文件"""
    timing_path = Path(TIMING_DIR)
    
    # 收集所有文件的所有记录，并标记来源
    all_records = []
    
    for json_file in timing_path.glob("timing_*.json"):
        records = parse_timing_file(json_file)
        for record in records:
            timestamp = get_record_time(record)
            if timestamp is not None:
                all_records.append({
                    'record': record,
                    'timestamp': timestamp,
                    'file': json_file.name,
                    'is_llm': is_llm_record(record)
                })
    
    if not all_records:
        print("没有找到任何记录")
        return
    
    # 按时间排序
    all_records.sort(key=lambda x: x['timestamp'])
    
    # 找到全局最早时间
    start_time = all_records[0]['timestamp']
    print(f"全局最早时间: {start_time}")
    print(f"对应记录: {all_records[0]['file']}")
    
    # 找到10分钟后的时间点
    ten_min_mark = start_time + TEN_MINUTES
    print(f"\n10分钟后的时间点: {ten_min_mark}")
    
    # 2小时结束时间
    two_hour_end = ten_min_mark + TWO_HOURS
    print(f"2小时后的结束时间点: {two_hour_end}")
    
    # 统计10分钟后到2小时内的LLM步骤
    llm_steps_in_window = 0
    total_steps_in_window = 0
    
    for item in all_records:
        ts = item['timestamp']
        if ten_min_mark <= ts <= two_hour_end:
            total_steps_in_window += 1
            if item['is_llm']:
                llm_steps_in_window += 1
    
    print(f"\n=== 统计结果 ===")
    print(f"从10分钟后开始的2小时窗口内:")
    print(f"  - LLM调用步骤数: {llm_steps_in_window}")
    print(f"  - 总步骤数 (包括tool): {total_steps_in_window}")
    
    # 额外统计：按文件分组的LLM步骤
    print(f"\n=== 按文件分组统计 (10分钟后的2小时内) ===")
    file_llm_counts = defaultdict(int)
    for item in all_records:
        ts = item['timestamp']
        if ten_min_mark <= ts <= two_hour_end and item['is_llm']:
            file_llm_counts[item['file']] += 1
    
    for fname, count in sorted(file_llm_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {fname}: {count} LLM steps")
    
    if len(file_llm_counts) > 20:
        print(f"  ... 还有 {len(file_llm_counts) - 20} 个文件")


if __name__ == "__main__":
    analyze_all_files()

