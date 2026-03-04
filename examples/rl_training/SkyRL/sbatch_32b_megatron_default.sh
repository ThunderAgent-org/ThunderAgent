#!/bin/bash
#SBATCH --job-name=32b-rdef
#SBATCH --partition=batch
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=176
#SBATCH --mem=0
#SBATCH --time=96:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 32B Megatron reproducibility job: ThunderAgent router=default.
# You may need to edit SBATCH headers (partition/account/time/cpus) for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPRO_ROUTER_MODE="default"
export REPRO_ROUTER_TAG="default"
unset REPRO_ACTING_TOKEN_WEIGHT || true

bash "$SCRIPT_DIR/scripts/repro/run_32b_megatron_repro_job.sh"
