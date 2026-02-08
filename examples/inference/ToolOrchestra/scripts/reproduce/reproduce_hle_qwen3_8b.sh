#!/usr/bin/env bash
set -euo pipefail

# Reproduce ToolOrchestra (HLE) throughput curves for Qwen3-8B.
# Runs C in {24,32,40,48} for baseline / continuum / thunderagent and
# collects windowed metrics (10–130min by default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f "${REPO_DIR}/setup_envs.sh" ]]; then
  echo "ERROR: setup_envs.sh not found. Copy setup_envs.sh_example and fill in paths/keys." >&2
  exit 1
fi

set +u
source "${REPO_DIR}/setup_envs.sh"
set -u

if [[ -z "${CKPT_DIR:-}" || -z "${INDEX_DIR:-}" || -z "${THUNDERAGENT_ROOT:-}" ]]; then
  echo "ERROR: CKPT_DIR, INDEX_DIR, and THUNDERAGENT_ROOT must be set (see setup_envs.sh)." >&2
  exit 1
fi

METHODS=(${METHODS:-baseline continuum thunderagent})
C_LIST=(${C_LIST:-24 32 40 48})
REP="${REP:-1}"

EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-150}"
WINDOW_START_SEC="${WINDOW_START_SEC:-600}"
WINDOW_END_SEC="${WINDOW_END_SEC:-7800}"
EXAMPLE_PATH="${EXAMPLE_PATH:-}"
EXAMPLE_ARG=()
if [[ -n "${EXAMPLE_PATH}" ]]; then
  EXAMPLE_ARG=(--example-path "${EXAMPLE_PATH}")
fi

OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/evaluation/outputs}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/evaluation/logs}"

kill_ports() {
  local ports=(8100 8000 1401)
  for p in "${ports[@]}"; do
    local pids
    pids="$(lsof -tiTCP:${p} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "[cleanup] killing listeners on port ${p}: ${pids}"
      kill ${pids} >/dev/null 2>&1 || true
      sleep 0.2
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  done
}

for method in "${METHODS[@]}"; do
  for c in "${C_LIST[@]}"; do
    kill_ports
    ts="$(date +%Y%m%d_%H%M%S)"
    out_dir="${OUT_ROOT}/hle_paper_${method}_c${c}_rep${REP}_${ts}"
    log_dir="${LOG_ROOT}/hle_paper_${method}_c${c}_rep${REP}_${ts}"
    echo "==== method=${method} C=${c} rep=${REP} ===="
    bash "${REPO_DIR}/evaluation/launch_hle_inference.sh" \
      --method "${method}" \
      --concurrency "${c}" \
      --eval-timeout-min "${EVAL_TIMEOUT_MIN}" \
      --window-start-sec "${WINDOW_START_SEC}" \
      --window-end-sec "${WINDOW_END_SEC}" \
      --output-dir "${out_dir}" \
      --log-dir "${log_dir}" \
      --thunderagent-root "${THUNDERAGENT_ROOT}" \
      "${EXAMPLE_ARG[@]}"
    echo "done: ${out_dir}"
  done
done
