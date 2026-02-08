# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for HLE evaluation profiling.

This is intentionally modeled after tau2-bench's logging_config.py so we can:
- use the same custom log levels (PROFILE=15, USER_JUDGE=16)
- emit lightweight, structured key=value logs
- parse logs post-run to generate timing/length histograms
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

# Custom log levels (between DEBUG=10 and INFO=20)
PROFILE = 15
USER_JUDGE = 16

logging.addLevelName(PROFILE, "PROFILE")
logging.addLevelName(USER_JUDGE, "USER_JUDGE")


class TaskContext(threading.local):
    """Thread-local storage for task execution context."""

    def __init__(self):
        super().__init__()
        self.task_id: Optional[str] = None
        self.step: Optional[int] = None
        self.domain: Optional[str] = None
        self.eid: Optional[int] = None


_task_context = TaskContext()


def set_task_context(task_id: str, domain: Optional[str] = None, eid: Optional[int] = None) -> None:
    _task_context.task_id = task_id
    _task_context.domain = domain
    _task_context.eid = eid
    _task_context.step = None


def set_step_context(step: Optional[int]) -> None:
    _task_context.step = step


def clear_task_context() -> None:
    _task_context.task_id = None
    _task_context.step = None
    _task_context.domain = None
    _task_context.eid = None


def get_task_context() -> Dict[str, Any]:
    return {
        "task_id": getattr(_task_context, "task_id", None),
        "step": getattr(_task_context, "step", None),
        "domain": getattr(_task_context, "domain", None),
        "eid": getattr(_task_context, "eid", None),
    }


@contextmanager
def task_context(task_id: str, domain: Optional[str] = None, eid: Optional[int] = None):
    set_task_context(task_id=task_id, domain=domain, eid=eid)
    try:
        yield
    finally:
        clear_task_context()


class HLEFormatter(logging.Formatter):
    """Formatter that adds millisecond timestamps + thread id + task context."""

    def format(self, record: logging.LogRecord) -> str:
        ct = datetime.fromtimestamp(record.created)
        record.timestamp_ms = ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(ct.microsecond / 1000):03d}"
        record.thread_id = threading.current_thread().ident

        ctx = get_task_context()
        record.task_id = ctx.get("task_id") or "global"
        record.step_num = ctx.get("step")
        record.domain = ctx.get("domain")
        record.eid = ctx.get("eid")
        return super().format(record)


class HLELogger(logging.Logger):
    def profile(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(PROFILE):
            self._log(PROFILE, msg, args, **kwargs)

    def user_judge(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(USER_JUDGE):
            self._log(USER_JUDGE, msg, args, **kwargs)


def _ensure_logger_class() -> None:
    # Best-effort: only set the logger class once for this process.
    # If something else already set a custom class, it's still usually compatible,
    # but we prefer to set ours early from eval entrypoints.
    try:
        if logging.getLoggerClass() is not HLELogger:
            logging.setLoggerClass(HLELogger)
    except Exception:
        pass


def get_hle_logger(name: str = "hle") -> HLELogger:
    _ensure_logger_class()
    return logging.getLogger(name)  # type: ignore[return-value]


def configure_hle_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    stream_handler: bool = True,
    file_handler: Optional[str] = None,
) -> HLELogger:
    level_map = {
        "DEBUG": logging.DEBUG,
        "PROFILE": PROFILE,
        "USER_JUDGE": USER_JUDGE,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = level_map.get((level or "INFO").upper(), logging.INFO)

    if format_string is None:
        format_string = "[%(levelname)s] %(timestamp_ms)s task=%(task_id)s thread=%(thread_id)s %(message)s"

    formatter = HLEFormatter(format_string)

    logger = get_hle_logger()
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Remove existing handlers to prevent duplicates if configure is called multiple times.
    logger.handlers.clear()

    if stream_handler:
        sh = logging.StreamHandler()
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if file_handler:
        fh = logging.FileHandler(file_handler)
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Timer:
    """Context manager measuring wall clock time in milliseconds."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - (self.start_time or self.end_time)) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000.0


def log_profile_event(event_type: str, **kwargs) -> None:
    """
    Log a PROFILE level event with structured key=value pairs.

    Common event types: llm_call, tool_call, task_complete
    """
    logger = get_hle_logger()
    ctx = get_task_context()

    parts = [f"type={event_type}"]

    if ctx.get("step") is not None:
        parts.append(f"step={ctx['step']}")
    if ctx.get("eid") is not None:
        parts.append(f"eid={ctx['eid']}")

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")

    logger.profile(" ".join(parts))


def log_user_judge_event(event_type: str, **kwargs) -> None:
    logger = get_hle_logger()
    ctx = get_task_context()

    parts = [f"type={event_type}"]

    if ctx.get("step") is not None:
        parts.append(f"step={ctx['step']}")
    if ctx.get("eid") is not None:
        parts.append(f"eid={ctx['eid']}")

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")

    logger.user_judge(" ".join(parts))


