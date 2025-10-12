from __future__ import annotations

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory
from sweagent.utils.step_sync import StepSynchronizer


class StepSyncAgentHook(AbstractAgentHook):
    """Enforces that all active instances stay on the same step within a batch."""

    def __init__(self, instance_id: str, synchronizer: StepSynchronizer):
        self._instance_id = instance_id
        self._synchronizer = synchronizer
        self._step_index = 0

    def on_run_start(self) -> None:
        self._synchronizer.on_instance_start(self._instance_id)

    def on_step_start(self):
        self._step_index += 1
        self._synchronizer.mark_step(self._instance_id, self._step_index)

    def on_step_done(self, *, step: StepOutput, info: AgentInfo):
        # nothing else to do per-step completion
        pass

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        self._synchronizer.mark_done(self._instance_id)

    def on_setup_attempt(self):
        self._step_index = 0
        self._synchronizer.reset_instance(self._instance_id)

    def on_tools_installation_started(self):
        # ensure environment preparation doesn't count as a step
        pass
