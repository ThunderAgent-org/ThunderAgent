#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv --python 3.12
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"

cd "${ROOT_DIR}/OpenHands"
uv pip install -e .
uv pip install datasets
uv pip install vllm==0.10.2 --torch-backend=auto
