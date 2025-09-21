"""LangGraph nodes implementing the SWE-Fixer pipeline steps."""

from __future__ import annotations

import json
import os
from typing import Optional

from pipeline.pipeline import code_edit, init_tokenizer, retrieve_files
from utils.file_operation import load_input_file
from utils.oai import load_openai_client

from .state import (
    EditingNodeInput,
    EditingState,
    PrepareNodeInput,
    PipelineContext,
    RetrievalNodeInput,
    RetrievalState,
    SweFixerState,
)
from langgraph.runtime import Runtime


def prepare_environment(
    state: PrepareNodeInput, runtime: Runtime[PipelineContext]
) -> SweFixerState:
    """Initialise tokenizer, client, and base datasets."""

    config = runtime.context["config"]
    config.validate()

    init_tokenizer(config.tokenizer_path)
    client = load_openai_client(config.api_key, config.base_url)

    instance_records = load_input_file(config.instance_file_path)
    instance_index = {
        record["instance_id"]: record for record in instance_records if "instance_id" in record
    }

    retrieval_state: RetrievalState = {
        "dataset_path": config.bm25_input_dataset,
        "result_path": config.bm25_results_file,
        **(state.get("retrieval", {})),
    }
    editing_state: EditingState = {
        "dataset_path": config.editing_dataset_path,
        "result_path": config.editing_result_path,
        **(state.get("editing", {})),
    }

    return {
        "client": client,
        "instance_index": instance_index,
        "retrieval": retrieval_state,
        "editing": editing_state,
    }


def run_retrieval(
    state: RetrievalNodeInput, runtime: Runtime[PipelineContext]
) -> SweFixerState:
    """Execute the retrieval portion of the pipeline if requested."""

    config = runtime.context["config"]
    if not config.run_retrieval:
        return {}

    client = state.get("client")
    if client is None:
        raise RuntimeError("Client must be prepared before running retrieval")

    try:
        retrieval_return_data = retrieve_files(
            client=client,
            input_file=config.bm25_retrieval_files,
            save_dataset_file=config.bm25_input_dataset,
            output_file=config.bm25_results_file,
            post_process=config.post_process,
        )
    except Exception as exc: 
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    retrieval_state: RetrievalState = {
        "handled_instances": retrieval_return_data,
        "dataset_path": config.bm25_input_dataset,
        "result_path": config.bm25_results_file,
    }

    return {"retrieval": retrieval_state}


def run_editing(
    state: EditingNodeInput, runtime: Runtime[PipelineContext]
) -> SweFixerState:
    """Execute the editing portion of the pipeline if requested."""

    config = runtime.context["config"]
    if not config.run_editing:
        return {}

    client = state.get("client")
    instance_index = state.get("instance_index")
    if client is None or instance_index is None:
        raise RuntimeError("Environment preparation must precede editing")

    try:
        code_edit(
            client=client,
            model_name=config.model_name,
            instance_file=instance_index,
            code_structure_path=config.code_structure_path,
            bm25_res_path=config.bm25_results_file,
            dataset_save_path=config.editing_dataset_path,
            output_file=config.editing_result_path,
            post_process=config.post_process,
        )
    except Exception as exc:  # pragma: no cover - surface upstream
        raise RuntimeError(f"Editing failed: {exc}") from exc

    editing_state: EditingState = {
        "dataset_path": config.editing_dataset_path,
        "result_path": config.editing_result_path,
    }

    return {"editing": editing_state}


