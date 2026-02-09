#!/bin/bash
# Common launcher for DP=4 SkyRL + ThunderAgent reproducibility jobs.
# Expected to be invoked by sbatch scripts with:
#   REPRO_ROUTER_MODE=default|tr
#   REPRO_ROUTER_TAG=default|tr_atw01
# Optional:
#   REPRO_ACTING_TOKEN_WEIGHT=0.1
#   SKYRL_HOST_DIR=/path/to/SkyRL
#   THUNDERAGENT_HOST_DIR=/path/to/ThunderAgent-wl
#   TRAINING_DATA_SYNC_DEST=/path/to/output/training_data
#   DOCKER_IMAGE=novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8

set -euo pipefail

if [ -z "${REPRO_ROUTER_MODE:-}" ]; then
  echo "ERROR: REPRO_ROUTER_MODE is required (default or tr)."
  exit 1
fi

if [ -z "${REPRO_ROUTER_TAG:-}" ]; then
  echo "ERROR: REPRO_ROUTER_TAG is required (for run_name and output paths)."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DEFAULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SKYRL_HOST_DIR="${SKYRL_HOST_DIR:-$SKYRL_DEFAULT_DIR}"

# In the target layout, SkyRL is inside: ThunderAgent-wl/examples/rl_training/SkyRL
# So ThunderAgent root is typically 3 levels above SkyRL.
THUNDERAGENT_DEFAULT_DIR="$(cd "$SKYRL_HOST_DIR/../../.." 2>/dev/null && pwd || true)"
THUNDERAGENT_HOST_DIR="${THUNDERAGENT_HOST_DIR:-$THUNDERAGENT_DEFAULT_DIR}"

TRAINING_DATA_SYNC_DEST="${TRAINING_DATA_SYNC_DEST:-$SKYRL_HOST_DIR/training_data}"
DOCKER_IMAGE="${DOCKER_IMAGE:-novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8}"

if [ ! -f "$SKYRL_HOST_DIR/launch_profiled_training_14b_dp4.sh" ]; then
  echo "ERROR: launch script not found at $SKYRL_HOST_DIR/launch_profiled_training_14b_dp4.sh"
  echo "Set SKYRL_HOST_DIR to your SkyRL repo path."
  exit 1
fi

if [ ! -f "$SKYRL_HOST_DIR/api-keys.sh" ]; then
  echo "ERROR: missing $SKYRL_HOST_DIR/api-keys.sh"
  echo "Create api-keys.sh with your WANDB_API_KEY (and optional HF_TOKEN)."
  exit 1
fi

if [ ! -d "$THUNDERAGENT_HOST_DIR" ]; then
  echo "ERROR: THUNDERAGENT_HOST_DIR does not exist: $THUNDERAGENT_HOST_DIR"
  echo "Set THUNDERAGENT_HOST_DIR to your ThunderAgent-wl repo path."
  exit 1
fi

NODE=$(hostname)
SCRATCH_DIR="/scratch/$USER"
ROUTER_MODE="$REPRO_ROUTER_MODE"
ROUTER_TAG="$REPRO_ROUTER_TAG"
ACTING_TOKEN_WEIGHT="${REPRO_ACTING_TOKEN_WEIGHT:-}"
SLURM_JOB_ID_SAFE="${SLURM_JOB_ID:-manual}"

if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  echo "=== Job $SLURM_JOB_ID_SAFE on $NODE, DP=4 router=$ROUTER_MODE acting_token_weight=$ACTING_TOKEN_WEIGHT ==="
else
  echo "=== Job $SLURM_JOB_ID_SAFE on $NODE, DP=4 router=$ROUTER_MODE ==="
fi

echo "=== SkyRL host dir: $SKYRL_HOST_DIR ==="
echo "=== ThunderAgent host dir: $THUNDERAGENT_HOST_DIR ==="
echo "=== Training data sync dest: $TRAINING_DATA_SYNC_DEST ==="
echo "=== Docker image: $DOCKER_IMAGE ==="
echo "=== $(date) ==="

# --- Step 0a: GPU cleanliness check ---
echo "=== Checking GPU memory... ==="
GPU_MEM_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
echo "$GPU_MEM_USED"
DIRTY_GPUS=$(echo "$GPU_MEM_USED" | awk -F', ' '$2 > 100 {print $1}')
if [ -n "$DIRTY_GPUS" ]; then
  echo "ERROR: GPUs not clean! The following GPUs have >100 MiB memory in use:"
  echo "$DIRTY_GPUS" | while read -r gpu; do echo "  GPU $gpu"; done
  echo "Aborting to avoid interference. Re-queue on a clean node."
  exit 1
fi
echo "=== All GPUs clean ==="

# --- Step 0b: Scratch directory ---
mkdir -p "$SCRATCH_DIR"
chmod 777 "$SCRATCH_DIR"

# --- Step 1: Docker setup + move data-root to /scratch ---
echo "=== Setting up Docker... ==="
sudo usermod -aG docker "$USER" 2>/dev/null || true

DOCKER_ROOT=/scratch/docker
if ! grep -q "$DOCKER_ROOT" /etc/docker/daemon.json 2>/dev/null; then
  echo "Configuring Docker data-root to $DOCKER_ROOT..."
  sudo systemctl stop docker 2>/dev/null || true
  sudo mkdir -p "$DOCKER_ROOT"
  if [ -f /etc/docker/daemon.json ]; then
    sudo python3 -c "
import json
with open('/etc/docker/daemon.json') as f:
    cfg = json.load(f)
cfg['data-root'] = '$DOCKER_ROOT'
with open('/etc/docker/daemon.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
  else
    echo "{\"data-root\": \"$DOCKER_ROOT\"}" | sudo tee /etc/docker/daemon.json > /dev/null
  fi
  sudo systemctl start docker
  echo "Docker data-root configured to $DOCKER_ROOT"
else
  echo "Docker data-root already configured to $DOCKER_ROOT"
fi

DOCKER_GID=$(getent group docker | cut -d: -f3)

# Verify image is usable (storage may have corrupted layers)
echo "=== Verifying Docker image... ==="
if ! sg docker -c "docker run --rm $DOCKER_IMAGE true" 2>/dev/null; then
  echo "Image corrupted or missing - wiping Docker storage and re-pulling..."
  sudo systemctl stop docker
  sudo rm -rf "$DOCKER_ROOT"
  sudo mkdir -p "$DOCKER_ROOT"
  sudo systemctl start docker
  sleep 3
  sg docker -c "docker pull $DOCKER_IMAGE"
  echo "Fresh Docker image pulled"
else
  echo "Docker image OK"
fi

sg docker -c "docker rm -f skyrl-head 2>/dev/null || true"
sg docker -c "docker ps -a --filter name=minisweagent -q | xargs -r docker rm -f 2>/dev/null || true"

sg docker -c "docker run -d --gpus all --cpuset-cpus=0-167 --shm-size=16g \
  --group-add $DOCKER_GID \
  --network=host --name skyrl-head \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v $SKYRL_HOST_DIR:/workspace/SkyRL \
  -v $THUNDERAGENT_HOST_DIR:/workspace/ThunderAgent-wl \
  -v $SCRATCH_DIR:/scratch \
  -e RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
  -e DATA=/scratch -e HOME=/scratch \
  -e UV_CACHE_DIR=/scratch/uv_cache -e UV_PYTHON_INSTALL_DIR=/scratch/uv_python \
  -e HF_HOME=/scratch/hf_cache -e HUGGINGFACE_HUB_CACHE=/scratch/hf_cache \
  -e FLASHINFER_WORKSPACE_DIR=/scratch/flashinfer_cache \
  -e TRITON_CACHE_DIR=/scratch/triton_cache \
  -e XDG_CACHE_HOME=/scratch/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $DOCKER_IMAGE \
  bash -c 'ray start --head --port=6379 --dashboard-host=0.0.0.0 && sleep infinity'"

sleep 5
echo "=== Ray status ==="
sg docker -c "docker exec skyrl-head ray status"

# --- Step 2: Sync code ---
echo "=== Syncing code to /scratch... ==="
sg docker -c "docker exec skyrl-head bash -c 'rsync -a --delete --exclude training_data --exclude slurm-\\* --exclude .git --exclude __pycache__ --exclude .claude /workspace/SkyRL/ /scratch/SkyRL/ && rsync -a --delete --exclude .git --exclude __pycache__ /workspace/ThunderAgent-wl/ /scratch/ThunderAgent-wl/'"

# --- Step 3: Prepare data ---
echo "=== Preparing data... ==="
sg docker -c "docker exec skyrl-head bash -c 'export HF_HOME=/scratch/hf_cache && cd /scratch/SkyRL/skyrl-train && uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir /scratch/data/swe_gym_subset'"

# --- Step 3.5: Pre-pull SWE-bench evaluation Docker images ---
echo "=== Pre-pulling SWE-bench images... ==="
sg docker -c "docker exec skyrl-head bash -c 'cd /scratch/SkyRL/skyrl-train && uv run --isolated examples/mini_swe_agent/pull_swebench_images.py --data_dir /scratch/data/swe_gym_subset --parallel 4 --train_only'"
echo "=== SWE-bench image pull complete ==="

# --- Step 4: Background NFS sync ---
JOB_SYNC_DEST="$TRAINING_DATA_SYNC_DEST/job_${SLURM_JOB_ID_SAFE}_dp4_${ROUTER_TAG}"
mkdir -p "$JOB_SYNC_DEST"
(
  while true; do
    sleep 60
    for d in "$SCRATCH_DIR/mini_swe_agent_trajs" "$SCRATCH_DIR/profiler_data" "$SCRATCH_DIR/thunderagent_profiles"; do
      [ -d "$d" ] && rsync -a --ignore-errors "$d" "$JOB_SYNC_DEST/" 2>/dev/null
    done
  done
) &
SYNC_PID=$!
echo "=== Background sync started (PID=$SYNC_PID) -> $JOB_SYNC_DEST ==="

# --- Step 5: Launch training ---
RUN_NAME="mini_swe_14B_dp4_${ROUTER_TAG}"
DOCKER_ENV_ARGS="-e THUNDERAGENT_ROUTER=$ROUTER_MODE -e SLURM_JOB_ID=$SLURM_JOB_ID_SAFE"
if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e ACTING_TOKEN_WEIGHT=$ACTING_TOKEN_WEIGHT"
fi

if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  echo "=== Launching 14B DP=4 training with router=$ROUTER_MODE acting_token_weight=$ACTING_TOKEN_WEIGHT ==="
else
  echo "=== Launching 14B DP=4 training with router=$ROUTER_MODE ==="
fi

set +e
sg docker -c "docker exec $DOCKER_ENV_ARGS skyrl-head bash /scratch/SkyRL/launch_profiled_training_14b_dp4.sh trainer.run_name=$RUN_NAME trainer.policy.record_memory=true"
TRAIN_EXIT=$?
set -e

# --- Cleanup ---
kill "$SYNC_PID" 2>/dev/null || true
echo "=== Background sync stopped ==="

echo "=== Job finished at $(date), exit=$TRAIN_EXIT ==="
exit "$TRAIN_EXIT"
