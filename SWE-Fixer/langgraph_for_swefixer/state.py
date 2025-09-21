"""State schemas for the LangGraph-backed SWE-Fixer pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from .config import PipelineConfig

class PipelineContext(TypedDict, total=False):
    config: PipelineConfig


class RetrievalState(TypedDict, total=False):
    handled_instances: Dict[str, List[str]]
    dataset_path: Optional[str]
    result_path: Optional[str]


class EditingState(TypedDict, total=False):
    dataset_path: Optional[str]
    result_path: Optional[str]


class PrepareNodeInput(TypedDict, total=False):
    retrieval: RetrievalState
    editing: EditingState


class RetrievalNodeInput(TypedDict, total=False):
    client: Any
    retrieval: RetrievalState


class EditingNodeInput(TypedDict, total=False):
    client: Any
    instance_index: Dict[str, Any]
    editing: EditingState



class SweFixerState(TypedDict, total=False):
    client: Any
    instance_index: Dict[str, Any]
    retrieval: RetrievalState
    editing: EditingState