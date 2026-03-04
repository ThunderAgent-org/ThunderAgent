#!/bin/bash
#SBATCH --job-name=32b-rtr
#SBATCH --partition=batch
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=176
#SBATCH --mem=0
#SBATCH --time=96:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 32B Megatron reproducibility job: ThunderAgent router=tr with acting_token_weight=1.0.
# You may need to edit SBATCH headers (partition/account/time/cpus) for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPRO_ROUTER_MODE="tr"
export REPRO_ROUTER_TAG="tr_atw10"
export REPRO_ACTING_TOKEN_WEIGHT="1.0"

bash "$SCRIPT_DIR/scripts/repro/run_32b_megatron_repro_job.sh"
