#!/usr/bin/env python3
"""
带磁盘监控的评估运行器

每 2 秒检测 overlay (/) 的可用空间，如果 <= 500GB 则终止所有进程。
用法: python run_with_disk_monitor.py --num-workers 24
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ============== 配置 ==============
DISK_THRESHOLD_GB = 500  # 磁盘空间阈值 (GB)
CHECK_INTERVAL = 2       # 检测间隔 (秒)
# =================================


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with disk monitoring")
    parser.add_argument("--num-workers", type=int, default=24,
                        help="Number of parallel evaluation workers (default: 24)")
    parser.add_argument("--monitor-csv", type=str, default=None,
                        help="CSV file name for system monitoring (default: system_monitor_{num_workers}.csv)")
    return parser.parse_args()

_shutdown_event = threading.Event()
_main_process: subprocess.Popen = None


def get_disk_available_gb() -> float:
    """获取 overlay (/) 的可用空间 (GB)"""
    usage = shutil.disk_usage('/')
    return usage.free / (1024 ** 3)


def monitor_disk_space():
    """磁盘监控线程：空间不足时触发关闭"""
    global _main_process
    
    while not _shutdown_event.is_set():
        try:
            available_gb = get_disk_available_gb()
            
            if available_gb <= DISK_THRESHOLD_GB:
                print(f"\n{'='*60}")
                print(f"⚠️  警告: overlay 可用空间不足!")
                print(f"⚠️  当前可用: {available_gb:.1f} GB <= 阈值 {DISK_THRESHOLD_GB} GB")
                print(f"⚠️  尝试执行 docker system prune -a --force ...")
                
                try:
                    subprocess.run(
                        ["docker", "system", "prune", "-a", "--force"], 
                        check=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL
                    )
                    print("✅ Docker 清理完成")
                except Exception as e:
                    print(f"❌ Docker 清理失败: {e}")
                
                available_gb = get_disk_available_gb()
                print(f"📊 清理后可用空间: {available_gb:.1f} GB")
                
                if available_gb <= DISK_THRESHOLD_GB:
                    print(f"⚠️  空间仍然不足，正在关闭所有进程...")
                    print(f"{'='*60}\n")
                    
                    _shutdown_event.set()
                    kill_process_tree(_main_process)
                    return
                else:
                    print(f"✅ 空间已恢复，继续运行")
                    print(f"{'='*60}\n")
        except Exception as e:
            print(f"[DiskMonitor] 错误: {e}")
        
        time.sleep(CHECK_INTERVAL)


def kill_process_tree(proc: subprocess.Popen):
    """终止进程及其所有子进程"""
    if proc is None or proc.poll() is not None:
        return
    
    try:
        # 尝试终止整个进程组
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    
    try:
        proc.terminate()
    except Exception:
        pass
    
    # 等待 3 秒后强制杀死
    time.sleep(3)
    
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.kill()
        except Exception:
            pass


def main() -> int:
    global _main_process
    
    args = parse_args()
    num_workers = args.num_workers
    monitor_csv = args.monitor_csv or f"system_monitor_{num_workers}.csv"
    
    # 打印启动信息
    available_gb = get_disk_available_gb()
    print(f"{'='*60}")
    print(f"🚀 启动带磁盘监控的评估程序")
    print(f"📊 当前 overlay 可用空间: {available_gb:.1f} GB")
    print(f"⚠️  阈值: {DISK_THRESHOLD_GB} GB (低于此值将自动终止)")
    print(f"⏱️  检测间隔: {CHECK_INTERVAL} 秒")
    print(f"👥 并行 workers: {num_workers}")
    print(f"📝 监控日志: {monitor_csv}")
    print(f"{'='*60}\n")
    
    # 启动磁盘监控线程
    monitor_thread = threading.Thread(target=monitor_disk_space, daemon=True)
    monitor_thread.start()
    
    # 启动原始评估脚本，使用新的进程组
    script_path = Path(__file__).resolve().parent / "run_swebench_eval.py"
    
    try:
        _main_process = subprocess.Popen(
            [sys.executable, str(script_path), 
             "--num-workers", str(num_workers),
             "--monitor-csv", monitor_csv],
            start_new_session=True,  # 创建新的进程组
        )
        
        # 等待进程完成，同时检查是否触发关闭
        while _main_process.poll() is None:
            if _shutdown_event.is_set():
                print("\n磁盘空间不足，评估已终止")
                return 1
            time.sleep(1)
        
        return _main_process.returncode or 0
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在清理...")
        _shutdown_event.set()
        kill_process_tree(_main_process)
        return 130
    except Exception as e:
        print(f"程序出错: {e}")
        _shutdown_event.set()
        kill_process_tree(_main_process)
        return 1
    finally:
        _shutdown_event.set()
        if _main_process and _main_process.poll() is None:
            kill_process_tree(_main_process)


if __name__ == "__main__":
    raise SystemExit(main())

