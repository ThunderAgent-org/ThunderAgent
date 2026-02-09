#!/bin/bash
#SBATCH --job-name=14b-dp4-rt01
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=176
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --oversubscribe
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# DP=4 reproducibility job: ThunderAgent router=tr, acting_token_weight=0.1.
# This explicit script avoids ambiguity around TR router defaults.
# You may need to edit SBATCH headers (partition/account/time) for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPRO_ROUTER_MODE="tr"
export REPRO_ROUTER_TAG="tr_atw01"
export REPRO_ACTING_TOKEN_WEIGHT="0.1"

bash "$SCRIPT_DIR/scripts/repro/run_dp4_repro_job.sh"
