"""Configuration objects for the LangGraph SWE-Fixer pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """User-supplied configuration for a SWE-Fixer run."""

    api_key: str
    base_url: str
    model_name: str
    tokenizer_path: str
    instance_file_path: str
    code_structure_path: str
    bm25_retrieval_files: str
    bm25_input_dataset: str
    bm25_results_file: str
    editing_dataset_path: str
    editing_result_path: str
    post_process: bool = False
    run_retrieval: bool = True
    run_editing: bool = True

    def require_retrieval_inputs(self) -> None:
        """Raise if retrieval inputs are missing when retrieval is enabled."""
        if not self.run_retrieval:
            return
        missing = [
            name
            for name, value in {
                "bm25_retrieval_files": self.bm25_retrieval_files,
                "bm25_input_dataset": self.bm25_input_dataset,
                "bm25_results_file": self.bm25_results_file,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Retrieval step requested but configuration missing: "
                + ", ".join(missing)
            )

    def require_editing_inputs(self) -> None:
        """Raise if editing inputs are missing when editing is enabled."""
        if not self.run_editing:
            return
        missing = [
            name
            for name, value in {
                "code_structure_path": self.code_structure_path,
                "bm25_results_file": self.bm25_results_file,
                "editing_dataset_path": self.editing_dataset_path,
                "editing_result_path": self.editing_result_path,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Editing step requested but configuration missing: "
                + ", ".join(missing)
            )

    def validate(self) -> None:
        """Ensure the configuration is internally consistent."""
        if not self.api_key:
            raise ValueError("api_key must be provided")
        if not self.base_url:
            raise ValueError("base_url must be provided")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must be provided")
        if not self.instance_file_path:
            raise ValueError("instance_file_path must be provided")
        if not self.model_name:
            raise ValueError("model_name must be provided")
        self.require_retrieval_inputs()
        self.require_editing_inputs()
