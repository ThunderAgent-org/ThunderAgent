#!/bin/bash
# Submit both DP=4 reproducibility jobs and print job IDs.
# Optional environment variables (inherited by sbatch):
#   SKYRL_HOST_DIR
#   THUNDERAGENT_HOST_DIR
#   TRAINING_DATA_SYNC_DEST
#   DOCKER_IMAGE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_SCRIPT="$SCRIPT_DIR/sbatch_repro_dp4_default.sh"
TR_SCRIPT="$SCRIPT_DIR/sbatch_repro_dp4_tr_atw01.sh"

if [ ! -f "$DEFAULT_SCRIPT" ] || [ ! -f "$TR_SCRIPT" ]; then
  echo "ERROR: reproducibility sbatch scripts are missing."
  exit 1
fi

echo "Submitting DP=4 default job..."
DEFAULT_OUT=$(sbatch --export=ALL "$DEFAULT_SCRIPT")
DEFAULT_JOB_ID=$(echo "$DEFAULT_OUT" | awk '{print $4}')
echo "$DEFAULT_OUT"

echo "Submitting DP=4 TR(atw=0.1) job..."
TR_OUT=$(sbatch --export=ALL "$TR_SCRIPT")
TR_JOB_ID=$(echo "$TR_OUT" | awk '{print $4}')
echo "$TR_OUT"

cat <<MSG

Submitted jobs:
- default: $DEFAULT_JOB_ID
- tr_atw01: $TR_JOB_ID

When at least one rollout step has completed in both jobs, compare tokens/sec with:
python3 $SCRIPT_DIR/analyze_dp4_tokens_compare.py \
  --default-job $DEFAULT_JOB_ID \
  --tr-job $TR_JOB_ID
MSG
