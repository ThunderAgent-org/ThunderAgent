#!/usr/bin/env bash
set -euo pipefail

# Run a single HLE setting with metrics collection.
# Defaults: thunderagent + concurrency=48 + 2h30 window (10–130min).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"

if [[ ! -f "${REPO_DIR}/setup_envs.sh" ]]; then
  echo "ERROR: setup_envs.sh not found. Copy setup_envs.sh_example and fill in paths/keys." >&2
  exit 1
fi

set +u
source "${REPO_DIR}/setup_envs.sh"
set -u

METHOD="${METHOD:-thunderagent}"
CONCURRENCY="${CONCURRENCY:-48}"
EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-150}"
WINDOW_START_SEC="${WINDOW_START_SEC:-600}"
WINDOW_END_SEC="${WINDOW_END_SEC:-7800}"
EXAMPLE_PATH="${EXAMPLE_PATH:-}"
EXAMPLE_ARG=()
if [[ -n "${EXAMPLE_PATH}" ]]; then
  EXAMPLE_ARG=(--example-path "${EXAMPLE_PATH}")
fi

if [[ -z "${THUNDERAGENT_ROOT:-}" ]]; then
  echo "ERROR: THUNDERAGENT_ROOT is not set (set it in setup_envs.sh)." >&2
  exit 1
fi

bash "${REPO_DIR}/evaluation/launch_hle_inference.sh" \
  --method "${METHOD}" \
  --concurrency "${CONCURRENCY}" \
  --eval-timeout-min "${EVAL_TIMEOUT_MIN}" \
  --window-start-sec "${WINDOW_START_SEC}" \
  --window-end-sec "${WINDOW_END_SEC}" \
  --thunderagent-root "${THUNDERAGENT_ROOT}" \
  "${EXAMPLE_ARG[@]}"
