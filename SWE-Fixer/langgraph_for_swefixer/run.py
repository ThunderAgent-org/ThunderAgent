from pathlib import Path

from .config import PipelineConfig
from .graph import run_swe_fixer_pipeline

ROOT = Path(__file__).resolve().parent

config = PipelineConfig(
    api_key=None,
    base_url=None,
    model_name="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    tokenizer_path="cl100k_base",
    instance_file_path="langgraph_for_swefixer/test_data/instance_file.jsonl",
    code_structure_path="langgraph_for_swefixer/test_data",
    bm25_retrieval_files="langgraph_for_swefixer/test_data/retrieval_data.jsonl",
    bm25_input_dataset="langgraph_for_swefixer/bm_input.json",
    bm25_results_file="langgraph_for_swefixer/bm_results.json",
    editing_dataset_path="langgraph_for_swefixer/edit_dataset.json",
    editing_result_path="langgraph_for_swefixer/edit_result.json"
)

run_swe_fixer_pipeline(config)
