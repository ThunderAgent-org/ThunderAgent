"""Graph orchestration for the LangGraph SWE-Fixer pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from .config import PipelineConfig
from .nodes import prepare_environment, run_editing, run_retrieval
from .state import PipelineContext, SweFixerState


def build_swe_fixer_app() -> Any:
    """Construct the LangGraph application for SWE-Fixer."""

    workflow = StateGraph(SweFixerState, context_schema=PipelineContext)
    workflow.add_node("prepare_environment", prepare_environment)
    workflow.add_node("run_retrieval", run_retrieval)
    workflow.add_node("run_editing", run_editing)

    workflow.set_entry_point("prepare_environment")
    workflow.add_edge("prepare_environment", "run_retrieval")
    workflow.add_edge("run_retrieval", "run_editing")
    workflow.add_edge("run_editing", END)

    return workflow.compile()


def run_swe_fixer_pipeline(
    config: PipelineConfig,
    *,
    initial_state: Optional[Dict[str, Any]] = None,
) -> SweFixerState:
    """Convenience helper to build and execute the pipeline."""

    app = build_swe_fixer_app()
    state: SweFixerState = {}
    if initial_state:
        state.update(initial_state)
    context: PipelineContext = {"config": config}
    return app.invoke(state, context=context)
