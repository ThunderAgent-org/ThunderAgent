#!/bin/bash
#SBATCH --job-name=14b-dp4-rdef
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

# DP=4 reproducibility job: ThunderAgent router=default.
# This script is intended for use in ThunderAgent/examples/rl_training style layouts.
# You may need to edit SBATCH headers (partition/account/time) for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPRO_ROUTER_MODE="default"
export REPRO_ROUTER_TAG="default"
unset REPRO_ACTING_TOKEN_WEIGHT || true

bash "$SCRIPT_DIR/scripts/repro/run_dp4_repro_job.sh"
