from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StageTimingLogger:
    def __init__(self, path: Path, *, instance_id: str):
        self._path = path
        self._instance_id = instance_id
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        stage: str,
        phase: str,
        step: int,
        attempt: int,
        thread_name: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "instance_id": self._instance_id,
            "thread": thread_name,
            "stage": stage,
            "phase": phase,
            "step": step,
            "attempt": attempt,
        }
        if extra:
            event["extra"] = extra
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
