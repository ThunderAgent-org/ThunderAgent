from __future__ import annotations

import threading
from typing import Any

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory
from sweagent.utils.stage_timing import StageTimingLogger


class StageTimingAgentHook(AbstractAgentHook):
    _STAGES = (
        "env_prepare",
        "llm_reasoning",
        "tool_execution",
        "observation_packaging",
    )

    def __init__(self, logger: StageTimingLogger):
        self._logger = logger
        self._step_index = 0
        self._attempt_index = 1
        self._stage_state: dict[str, dict[str, Any]] = {
            stage: {"active": False, "step": 0, "attempt": 1} for stage in self._STAGES
        }
        self._env_prepare_started = False
        self._env_prepared = False

    def on_run_start(self) -> None:
        self._step_index = 0
        self._attempt_index = 1
        self._env_prepare_started = False
        self._env_prepared = False
        self._reset_active_stages(reason="run_start")

    def on_setup_attempt(self) -> None:
        self._attempt_index += 1
        self._step_index = 0
        self._env_prepare_started = False
        self._env_prepared = False
        self._reset_active_stages(reason="setup_attempt")

    def on_step_start(self) -> None:
        self._reset_active_stages(reason="step_start")
        self._step_index += 1
        if not self._env_prepared and self._env_prepare_started:
            self.finish_env_prepare(extra={"reason": "step_start"})

    def on_model_query(self, *, messages, agent: str) -> None:  # type: ignore[override]
        if not self._env_prepared and self._env_prepare_started:
            self.finish_env_prepare(extra={"reason": "model_query"})
        self._enter_stage("llm_reasoning")

    def on_actions_generated(self, *, step: StepOutput) -> None:
        self._exit_stage("llm_reasoning")

    def on_action_started(self, *, step: StepOutput) -> None:
        self._enter_stage("tool_execution")

    def on_action_executed(self, *, step: StepOutput) -> None:
        self._exit_stage("tool_execution")
        self._enter_stage("observation_packaging")

    def on_step_done(self, *, step: StepOutput, info: AgentInfo) -> None:
        self._exit_stage("observation_packaging")
        self._reset_active_stages(reason="step_done")

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo) -> None:
        self._reset_active_stages(reason="run_done")

    def start_env_prepare(self, extra: dict[str, Any] | None = None) -> None:
        if self._env_prepare_started:
            return
        self._env_prepare_started = True
        state = self._stage_state["env_prepare"]
        state["step"] = 0
        state["attempt"] = self._attempt_index
        self._enter_stage("env_prepare", extra=extra)

    def finish_env_prepare(self, extra: dict[str, Any] | None = None) -> None:
        if not self._env_prepare_started or self._env_prepared:
            return
        self._exit_stage("env_prepare", extra=extra, force=True)
        self._env_prepared = True

    def _enter_stage(self, stage: str, extra: dict[str, Any] | None = None) -> None:
        state = self._stage_state[stage]
        if state["active"]:
            self._exit_stage(stage, extra={"reason": "reenter"}, force=True)
        state["active"] = True
        state["step"] = self._step_index
        state["attempt"] = self._attempt_index
        self._record(stage, "enter", state["step"], state["attempt"], extra)

    def _exit_stage(
        self,
        stage: str,
        extra: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None:
        state = self._stage_state[stage]
        if not state["active"]:
            if not force:
                return
            step = state["step"] or self._step_index
            attempt = state["attempt"]
        else:
            step = state["step"]
            attempt = state["attempt"]
        state["active"] = False
        self._record(stage, "exit", step, attempt, extra)

    def _reset_active_stages(self, *, reason: str) -> None:
        for stage, state in self._stage_state.items():
            if state["active"]:
                self._exit_stage(stage, extra={"reason": reason}, force=True)
                state["step"] = 0
                state["attempt"] = self._attempt_index

    def _record(self, stage: str, phase: str, step: int, attempt: int, extra: dict[str, Any] | None) -> None:
        thread_name = threading.current_thread().name
        if step <= 0:
            if stage == "env_prepare" and not self._env_prepared:
                step = 0
            else:
                step = self._step_index or 1
        self._logger.record(
            stage=stage,
            phase=phase,
            step=step,
            attempt=attempt,
            thread_name=thread_name,
            extra=extra,
        )
