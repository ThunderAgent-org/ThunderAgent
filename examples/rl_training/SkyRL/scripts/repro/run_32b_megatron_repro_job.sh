#!/bin/bash
# Common launcher for 32B Megatron fully async SkyRL + ThunderAgent reproducibility jobs.
# Expected to be invoked by sbatch scripts with:
#   REPRO_ROUTER_MODE=default|tr
#   REPRO_ROUTER_TAG=default|tr_atw10
# Optional:
#   REPRO_ACTING_TOKEN_WEIGHT=1.0
#   SKYRL_HOST_DIR=/path/to/SkyRL
#   THUNDERAGENT_HOST_DIR=/path/to/ThunderAgent-wl
#   TRAINING_DATA_SYNC_DEST=/path/to/output/training_data
#   DOCKER_IMAGE=novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8
#
# Architecture (6-node layout):
#   Node 0 (HEAD):     Ray GCS + 128 SWE workers + Docker-in-Docker (--num-gpus=0, 8 GPUs idle)
#   Node 1 (Inference): 4 vLLM TP=2 engines (8 GPUs)
#   Node 2-5 (Training): Megatron TP=4 PP=4 CP=2 (32 GPUs)
#   Total: 48 GPUs allocated by SLURM, 40 used

set -euo pipefail

# --- Validate required env vars ---
if [ -z "${REPRO_ROUTER_MODE:-}" ]; then
  echo "ERROR: REPRO_ROUTER_MODE is required (default or tr)."
  exit 1
fi

if [ -z "${REPRO_ROUTER_TAG:-}" ]; then
  echo "ERROR: REPRO_ROUTER_TAG is required (for run_name and output paths)."
  exit 1
fi

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DEFAULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SKYRL_HOST_DIR="${SKYRL_HOST_DIR:-$SKYRL_DEFAULT_DIR}"

# Source user's cluster config if present
[ -f "$SKYRL_HOST_DIR/cluster-config.sh" ] && source "$SKYRL_HOST_DIR/cluster-config.sh"

# Recompute dependent defaults after cluster-config overrides.
# This avoids stale THUNDERAGENT_HOST_DIR when cluster-config changes SKYRL_HOST_DIR.
SKYRL_HOST_DIR="${SKYRL_HOST_DIR:-$SKYRL_DEFAULT_DIR}"
THUNDERAGENT_DEFAULT_DIR="$(cd "$SKYRL_HOST_DIR/../../.." 2>/dev/null && pwd || true)"
THUNDERAGENT_HOST_DIR="${THUNDERAGENT_HOST_DIR:-$THUNDERAGENT_DEFAULT_DIR}"

TRAINING_DATA_SYNC_DEST="${TRAINING_DATA_SYNC_DEST:-$SKYRL_HOST_DIR/training_data}"
DOCKER_IMAGE="${DOCKER_IMAGE:-novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8}"

# NCCL cross-node config (critical for multi-node Docker training)
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens7}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# --- Validate ---
if [ ! -f "$SKYRL_HOST_DIR/launch_training_32b_megatron.sh" ]; then
  echo "ERROR: launch script not found at $SKYRL_HOST_DIR/launch_training_32b_megatron.sh"
  echo "Set SKYRL_HOST_DIR to your SkyRL repo path."
  exit 1
fi

if [ ! -f "$SKYRL_HOST_DIR/api-keys.sh" ]; then
  echo "ERROR: missing $SKYRL_HOST_DIR/api-keys.sh"
  echo "Create api-keys.sh with your WANDB_API_KEY and HF_TOKEN."
  echo "See api-keys.sh.example for the template."
  exit 1
fi

if [ ! -d "$THUNDERAGENT_HOST_DIR" ]; then
  echo "ERROR: THUNDERAGENT_HOST_DIR does not exist: $THUNDERAGENT_HOST_DIR"
  echo "Set THUNDERAGENT_HOST_DIR to your ThunderAgent repo path."
  exit 1
fi

# --- Node assignment ---
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
if [ ${#NODELIST[@]} -lt 6 ]; then
  echo "ERROR: Need 6 nodes, got ${#NODELIST[@]}. Use --nodes=6 in your sbatch script."
  exit 1
fi

HEAD_NODE=${NODELIST[0]}       # Ray GCS + Docker-in-Docker (--num-gpus=0, 8 GPUs idle)
INFERENCE_NODE=${NODELIST[1]}  # vLLM engines (8 GPUs)
TRAINING_NODE_0=${NODELIST[2]} # Megatron PP stage 0 (8 GPUs)
TRAINING_NODE_1=${NODELIST[3]} # Megatron PP stage 1 (8 GPUs)
TRAINING_NODE_2=${NODELIST[4]} # Megatron PP stage 2 (8 GPUs)
TRAINING_NODE_3=${NODELIST[5]} # Megatron PP stage 3 (8 GPUs)

SCRATCH_DIR="/scratch/$USER"
DOCKER_ROOT="/scratch/$USER/docker"
ROUTER_MODE="$REPRO_ROUTER_MODE"
ROUTER_TAG="$REPRO_ROUTER_TAG"
ACTING_TOKEN_WEIGHT="${REPRO_ACTING_TOKEN_WEIGHT:-}"
SLURM_JOB_ID_SAFE="${SLURM_JOB_ID:-manual}"

echo "=== Job $SLURM_JOB_ID_SAFE: 32B Megatron (6-node layout), router=$ROUTER_MODE ==="
echo "=== HEAD_NODE=$HEAD_NODE (rollout Docker, 0 GPUs registered) ==="
echo "=== INFERENCE_NODE=$INFERENCE_NODE (vLLM, 8 GPUs) ==="
echo "=== TRAINING_NODE_0=$TRAINING_NODE_0 (PP stage 0, 8 GPUs) ==="
echo "=== TRAINING_NODE_1=$TRAINING_NODE_1 (PP stage 1, 8 GPUs) ==="
echo "=== TRAINING_NODE_2=$TRAINING_NODE_2 (PP stage 2, 8 GPUs) ==="
echo "=== TRAINING_NODE_3=$TRAINING_NODE_3 (PP stage 3, 8 GPUs) ==="
echo "=== SkyRL host dir: $SKYRL_HOST_DIR ==="
echo "=== ThunderAgent host dir: $THUNDERAGENT_HOST_DIR ==="
echo "=== Training data sync dest: $TRAINING_DATA_SYNC_DEST ==="
echo "=== Docker image: $DOCKER_IMAGE ==="
echo "=== NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME, NCCL_IB_DISABLE=$NCCL_IB_DISABLE ==="
if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  echo "=== acting_token_weight=$ACTING_TOKEN_WEIGHT ==="
fi
echo "=== $(date) ==="

# srun helper: run on a specific node
run_on() {
    local NODE=$1; shift
    srun --overlap --nodes=1 --ntasks=1 --gpus=0 --nodelist=$NODE "$@"
}

# --- Step 0: GPU cleanliness check ---
echo "=== Checking GPU memory on GPU nodes ==="
for NODE in $INFERENCE_NODE $TRAINING_NODE_0 $TRAINING_NODE_1 $TRAINING_NODE_2 $TRAINING_NODE_3; do
    run_on $NODE nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F', ' -v node=$NODE '$2 > 100 {found=1; print "DIRTY GPU "$1" on "node} END {if(found) exit 1}'
done
echo "=== All GPUs clean ==="

# --- Step 1: Per-node setup ---
# The setup logic runs inline via heredoc to avoid temp file management.
# Each node: Docker cleanup -> container start -> Ray start -> code sync -> venv -> model download
setup_node() {
    local NODE=$1
    local ROLE=$2       # "head" or "worker"
    local CONTAINER=$3  # container name
    local RAY_CMD=$4    # ray start command

    run_on $NODE bash -s "$ROLE" "$CONTAINER" "$RAY_CMD" "$HEAD_NODE" "$DOCKER_IMAGE" \
        "$DOCKER_ROOT" "$SCRATCH_DIR" "$SKYRL_HOST_DIR" "$THUNDERAGENT_HOST_DIR" \
        "$NCCL_SOCKET_IFNAME" "$NCCL_IB_DISABLE" << 'NODE_SETUP_EOF'
#!/bin/bash
set -e
ROLE=$1
CONTAINER=$2
RAY_CMD=$3
HEAD_NODE=$4
DOCKER_IMAGE=$5
DOCKER_ROOT=$6
SCRATCH_DIR=$7
SKYRL_HOST_DIR=$8
THUNDERAGENT_HOST_DIR=$9

NCCL_SOCKET_IFNAME=${10}
NCCL_IB_DISABLE=${11}

DOCKER_GID=$(getent group docker | cut -d: -f3)

echo "=== Node $(hostname): Setup as $ROLE ==="

sudo usermod -aG docker $USER 2>/dev/null || true

# Verify Docker is running
sg docker -c "docker info" > /dev/null 2>&1 || {
    echo "Docker not responding, restarting..."
    sudo systemctl restart docker
    sleep 5
}

# Cleanup old containers
sg docker -c "docker rm -f $CONTAINER 2>/dev/null || true"
sg docker -c "docker ps -a --filter name=minisweagent -q | xargs -r docker rm -f 2>/dev/null || true"
sg docker -c "docker system prune -f 2>/dev/null || true"

# Verify image exists
if ! sg docker -c "docker image inspect $DOCKER_IMAGE" > /dev/null 2>&1; then
    echo "Image missing on $(hostname) - pulling..."
    sg docker -c "docker pull $DOCKER_IMAGE"
fi

# Extra env vars: DOCKER_NODE_RESOURCE only on head node
EXTRA_ENV=""
if [ "$ROLE" = "head" ]; then
    EXTRA_ENV="-e DOCKER_NODE_RESOURCE=1"
fi

# Read HF_TOKEN from api-keys.sh (sourced inside subshell to avoid polluting env)
HF_TOKEN=$(bash -c "source $SKYRL_HOST_DIR/api-keys.sh 2>/dev/null && echo \$HF_TOKEN" || echo "")

# Start container
# NCCL logs redirected to /scratch/nccl_logs/ to keep training output clean
sg docker -c "docker run -d --gpus all --cpuset-cpus=0-167 --shm-size=16g \
  --group-add $DOCKER_GID \
  --network=host --name $CONTAINER \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v $SKYRL_HOST_DIR:/workspace/SkyRL \
  -v $THUNDERAGENT_HOST_DIR:/workspace/ThunderAgent-wl \
  -v $SCRATCH_DIR:/scratch \
  -e RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
  -e RAY_task_events_report_worker_events=false \
  -e DATA=/scratch -e HOME=/scratch \
  -e UV_CACHE_DIR=/scratch/uv_cache -e UV_PYTHON_INSTALL_DIR=/scratch/uv_python \
  -e HF_HOME=/scratch/hf_cache -e HUGGINGFACE_HUB_CACHE=/scratch/hf_cache -e HF_TOKEN=$HF_TOKEN \
  -e FLASHINFER_WORKSPACE_DIR=/scratch/flashinfer_cache \
  -e TRITON_CACHE_DIR=/scratch/triton_cache \
  -e XDG_CACHE_HOME=/scratch/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,COLL \
  -e NCCL_DEBUG_FILE=/scratch/nccl_logs/nccl_%h_%p.log \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
  -e CPATH=/scratch/SkyRL/skyrl-train/.venv/lib/python3.12/site-packages/nvidia/cudnn/include:/scratch/SkyRL/skyrl-train/.venv/lib/python3.12/site-packages/nvidia/nccl/include \
  -e CUDA_HOME=/usr/local/cuda \
  -e NVTE_FRAMEWORK=pytorch -e NVTE_WITH_USERBUFFERS=1 \
  $EXTRA_ENV \
  $DOCKER_IMAGE \
  bash -c 'mkdir -p /scratch/nccl_logs && sleep infinity'"
echo "$(hostname): Container $CONTAINER started"

# Start Ray inside the container
sleep 2
sg docker -c "docker exec -i $CONTAINER bash" << RAYEOF
ray stop --force 2>/dev/null || true
$RAY_CMD
RAYEOF
echo "$(hostname): Ray started"

# Sync code inside container
sg docker -c "docker exec $CONTAINER bash -c '
  timeout 180 rsync -a --delete \
    --exclude training_data --exclude slurm-\\* --exclude .git --exclude __pycache__ \
    --exclude .claude --exclude .venv --exclude .tmp_\\* --exclude exp_\\* \
    --exclude \\*.png --exclude \\*.jpg --exclude docker/ --exclude training_data/ \
    --timeout=120 \
    /workspace/SkyRL/ /scratch/SkyRL/ 2>/dev/null
  RSYNC1=\$?
  timeout 60 rsync -a --delete --exclude .git --exclude __pycache__ --timeout=60 \
    /workspace/ThunderAgent-wl/ /scratch/ThunderAgent-wl/ 2>/dev/null
  RSYNC2=\$?
  if [ \$RSYNC1 -ne 0 ] || [ \$RSYNC2 -ne 0 ]; then
    echo \"WARNING: rsync incomplete (SkyRL=\$RSYNC1, ThunderAgent=\$RSYNC2), using pre-copied or cached code\"
  fi
  test -f /scratch/SkyRL/skyrl-train/skyrl_train/__init__.py && echo \"Code OK\" || echo \"ERROR: code missing\"
'" || echo "$(hostname): WARNING - rsync had issues, proceeding anyway"
echo "$(hostname): Code synced"

# Install venv if missing, verify cached environment
sg docker -c "docker exec $CONTAINER bash -c '
  cd /scratch/SkyRL/skyrl-train
  if ! .venv/bin/python --version > /dev/null 2>&1; then
    echo \"$(hostname): Venv missing - installing...\"
    uv sync --extra mcore --extra miniswe 2>&1 | tail -3
  fi
  echo \"Venv: \$(.venv/bin/python --version 2>&1 || echo MISSING)\"
  .venv/bin/python -c \"import megatron.core; print(f\\\"megatron-core {megatron.core.__version__}\\\")\" 2>&1 || echo \"megatron-core import FAILED\"

  # Download model if missing
  MODEL_SHARDS=\$(ls /scratch/models/Qwen3-32B/*.safetensors 2>/dev/null | wc -l)
  echo \"Model: \$MODEL_SHARDS shards\"
  if [ \"\$MODEL_SHARDS\" -lt 17 ]; then
    echo \"$(hostname): Model incomplete - downloading...\"
    mkdir -p /scratch/models/Qwen3-32B
    .venv/bin/huggingface-cli download Qwen/Qwen3-32B --local-dir /scratch/models/Qwen3-32B 2>&1 | tail -3
    echo \"Model: \$(ls /scratch/models/Qwen3-32B/*.safetensors 2>/dev/null | wc -l) shards (after download)\"
  fi

  echo \"Dataset: \$(ls /scratch/data/swe_gym/*.parquet 2>/dev/null | wc -l) parquets\"
'"

# Only on head node: symlink cuDNN/NCCL headers + check SWE-bench images
if [ "$ROLE" = "head" ]; then
    sg docker -c "docker exec -u root $CONTAINER bash -c '
      VENV=/scratch/SkyRL/skyrl-train/.venv/lib/python3.12/site-packages/nvidia
      for f in \$VENV/cudnn/include/cudnn*.h; do ln -sf \$f /usr/local/cuda/include/; done 2>/dev/null
      for f in \$VENV/nccl/include/nccl*.h; do ln -sf \$f /usr/local/cuda/include/; done 2>/dev/null
      for f in \$VENV/cudnn/lib/libcudnn*.so*; do ln -sf \$f /usr/local/cuda/lib64/; done 2>/dev/null
      for f in \$VENV/nccl/lib/libnccl*.so*; do ln -sf \$f /usr/local/cuda/lib64/; done 2>/dev/null
      echo \"cuDNN/NCCL headers symlinked\"
    '"
    SWEB_COUNT=$(sg docker -c "docker exec $CONTAINER bash -c 'docker images --format \"{{.Repository}}\" 2>/dev/null | grep -c sweb || echo 0'")
    echo "$(hostname): SWE-bench images cached: $SWEB_COUNT"
fi

echo "=== $(hostname) setup complete ==="
NODE_SETUP_EOF
}

# --- Step 2a: Setup HEAD node first (Ray head must be ready before workers) ---
echo "=== Setting up HEAD node first ==="
setup_node "$HEAD_NODE" head skyrl-head \
    "ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-gpus=0 --resources='{\"docker_node\": 128}'"
echo "=== Head node setup complete (Ray GCS ready on ${HEAD_NODE}:6379) ==="

# --- Step 2b: Setup 5 worker nodes in parallel (Ray head is guaranteed ready) ---
echo "=== Setting up 5 worker nodes in parallel ==="

setup_node "$INFERENCE_NODE" worker skyrl-worker \
    "ray start --address=${HEAD_NODE}:6379" &
INF_PID=$!

setup_node "$TRAINING_NODE_0" worker skyrl-worker \
    "ray start --address=${HEAD_NODE}:6379" &
TRAIN0_PID=$!

setup_node "$TRAINING_NODE_1" worker skyrl-worker \
    "ray start --address=${HEAD_NODE}:6379" &
TRAIN1_PID=$!

setup_node "$TRAINING_NODE_2" worker skyrl-worker \
    "ray start --address=${HEAD_NODE}:6379" &
TRAIN2_PID=$!

setup_node "$TRAINING_NODE_3" worker skyrl-worker \
    "ray start --address=${HEAD_NODE}:6379" &
TRAIN3_PID=$!

wait $INF_PID
echo "=== Inference node setup complete ==="
wait $TRAIN0_PID
echo "=== Training node PP0 setup complete ==="
wait $TRAIN1_PID
echo "=== Training node PP1 setup complete ==="
wait $TRAIN2_PID
echo "=== Training node PP2 setup complete ==="
wait $TRAIN3_PID
echo "=== Training node PP3 setup complete ==="

sleep 15
echo "=== Ray cluster status ==="
run_on $HEAD_NODE sg docker -c "docker exec skyrl-head ray status"

# --- Step 3: Pre-pull SWE-bench images in background ---
echo "=== Starting SWE-bench image pre-pull in background ==="
run_on $HEAD_NODE sg docker -c "docker exec skyrl-head bash -c '
  export HF_HOME=/scratch/hf_cache
  cd /scratch/SkyRL/skyrl-train
  EXISTING=\$(docker images --format \"{{.Repository}}\" 2>/dev/null | grep -c sweb || echo 0)
  echo \"Pre-pull: \$EXISTING SWE-bench images already cached\"
  if [ \$EXISTING -lt 2400 ]; then
    nohup uv run --isolated python /scratch/SkyRL/skyrl-train/examples/mini_swe_agent/pull_swebench_images.py \
      --train_dataset SumanthRH/SWE-Gym \
      --train_only \
      --parallel 8 \
      > /scratch/prepull_swebench.log 2>&1 &
    echo \"Pre-pull started (PID=\$!), log at /scratch/prepull_swebench.log\"
  else
    echo \"All images cached, skipping pre-pull\"
  fi
'" &
PREPULL_PID=$!
echo "=== SWE-bench pre-pull launched (host PID=$PREPULL_PID) ==="

# --- Step 4: Background NFS sync ---
JOB_SYNC_DEST="$TRAINING_DATA_SYNC_DEST/job_${SLURM_JOB_ID_SAFE}_32b_megatron_${ROUTER_TAG}"
mkdir -p "$JOB_SYNC_DEST"
(while true; do
    sleep 60
    for d in $SCRATCH_DIR/mini_swe_agent_trajs $SCRATCH_DIR/profiler_data $SCRATCH_DIR/thunderagent_profiles; do
        [ -d "$d" ] && rsync -a --ignore-errors "$d" "$JOB_SYNC_DEST/" 2>/dev/null
    done
done) &
SYNC_PID=$!
echo "=== Background sync started (PID=$SYNC_PID) -> $JOB_SYNC_DEST ==="

# --- Step 5: Launch training on HEAD node ---
RUN_NAME="mini_swe_32B_megatron_${ROUTER_TAG}"
DOCKER_ENV_ARGS="-e THUNDERAGENT_ROUTER=$ROUTER_MODE -e SLURM_JOB_ID=$SLURM_JOB_ID_SAFE"
DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e HEAD_NODE=$HEAD_NODE -e DOCKER_NODE_RESOURCE=1"
if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e ACTING_TOKEN_WEIGHT=$ACTING_TOKEN_WEIGHT"
fi

if [ -n "$ACTING_TOKEN_WEIGHT" ]; then
  echo "=== Launching 32B Megatron training with router=$ROUTER_MODE acting_token_weight=$ACTING_TOKEN_WEIGHT ==="
else
  echo "=== Launching 32B Megatron training with router=$ROUTER_MODE ==="
fi
echo "=== $(date) ==="

set +e
run_on $HEAD_NODE sg docker -c "docker exec $DOCKER_ENV_ARGS skyrl-head \
  bash /scratch/SkyRL/launch_training_32b_megatron.sh \
  trainer.run_name=$RUN_NAME \
  trainer.policy.record_memory=true \
  trainer.ckpt_interval=10"
TRAIN_EXIT=$?
set -e

# --- Cleanup ---
kill "$SYNC_PID" 2>/dev/null || true
echo "=== Background sync stopped ==="

echo "=== Job finished at $(date), exit=$TRAIN_EXIT ==="
exit "$TRAIN_EXIT"
