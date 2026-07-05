#!/bin/bash
# Submit both 32B Megatron repro jobs (default vs TR router) and print job IDs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_DEFAULT=$(sbatch --export=ALL --parsable "$SCRIPT_DIR/sbatch_32b_megatron_default.sh")
JOB_TR=$(sbatch --export=ALL --parsable "$SCRIPT_DIR/sbatch_32b_megatron_tr.sh")

echo "Submitted:"
echo "  default: $JOB_DEFAULT"
echo "  tr:      $JOB_TR"
