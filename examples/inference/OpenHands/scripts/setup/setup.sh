#!/usr/bin/env bash
set -euo pipefail

# NOTE: If you want the venv activation to affect your current shell, run:
#   source setup.sh

# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install OpenHands (code) in editable mode
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/OpenHands

