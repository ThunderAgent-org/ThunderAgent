#!/usr/bin/env bash
set -euo pipefail

# NOTE: If you want the venv activation to affect your current shell, run:
#   source setup.sh

# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM (GPU build), mini-swe-agent in editable mode, and datasets
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/mini-swe-agent
uv pip install datasets

