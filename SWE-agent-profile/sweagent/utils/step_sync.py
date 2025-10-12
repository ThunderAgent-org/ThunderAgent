from __future__ import annotations

from collections.abc import Iterable
from threading import Condition, Lock


class StepSynchronizer:
    """Synchronize agent steps across multiple instances in the same batch."""

    def __init__(self, instance_ids: Iterable[str]):
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._active = set(instance_ids)
        self._steps = {instance_id: 0 for instance_id in self._active}

    def on_instance_start(self, instance_id: str) -> None:
        with self._condition:
            if instance_id not in self._active:
                self._active.add(instance_id)
            self._steps.setdefault(instance_id, 0)
            self._condition.notify_all()

    def reset_instance(self, instance_id: str) -> None:
        with self._condition:
            if instance_id in self._active:
                self._steps[instance_id] = 0
                self._condition.notify_all()

    def mark_step(self, instance_id: str, step: int) -> None:
        with self._condition:
            if instance_id not in self._active:
                return
            self._steps[instance_id] = step
            self._condition.notify_all()
            while any(self._steps.get(other, 0) < step for other in self._active):
                self._condition.wait()

    def mark_done(self, instance_id: str) -> None:
        with self._condition:
            self._active.discard(instance_id)
            self._steps.pop(instance_id, None)
            self._condition.notify_all()
