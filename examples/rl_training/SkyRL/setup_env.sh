#!/bin/bash
# setup_env.sh - Automates the setup steps
# args: <JOBID>

export JOBID=${1:-25363}
export NODE1=research-secure-10
export NODE2=research-secure-11
export SCRATCH_DIR="/scratch/$USER"

echo "Using JOBID=$JOBID, NODE1=$NODE1, NODE2=$NODE2"
echo "Scratch dir: $SCRATCH_DIR"

echo "=== Step 0: Ensure scratch directory exists ==="
srun --jobid=$JOBID --nodes=2 --ntasks=2 bash -c "mkdir -p $SCRATCH_DIR && chmod 777 $SCRATCH_DIR && echo \"Created $SCRATCH_DIR on \$(hostname)\""

echo "=== Step 0.5: Cleanup existing containers ==="
srun --jobid=$JOBID --nodes=2 --ntasks=2 bash -c 'newgrp docker << EOF
docker rm -f skyrl-head skyrl-worker 2>/dev/null || true
EOF'

echo "=== Step 1: Docker group setup ==="
srun --jobid=$JOBID --nodes=2 --ntasks=2 bash -c 'sudo usermod -aG docker $USER' || true

echo "=== Step 2: Pull image ==="
srun --jobid=$JOBID --nodes=2 --ntasks=2 bash -c 'newgrp docker << EOF
docker pull novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8
EOF'

echo "=== Step 3: Start head node ==="
srun --overlap --jobid=$JOBID --nodes=1 --nodelist=$NODE1 bash -c "DOCKER_GID=\$(getent group docker | cut -d: -f3) && newgrp docker << EOF
docker rm -f skyrl-head 2>/dev/null
docker run -d --runtime=nvidia --gpus all --cpuset-cpus=0-167 --shm-size=16g \
  --group-add \$DOCKER_GID \
  --network=host --name skyrl-head \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v /home/hkang/wl/SkyRL:/workspace/SkyRL \
  -v $SCRATCH_DIR:/scratch \
  -e RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
  -e DATA=/scratch -e HOME=/scratch \
  -e UV_CACHE_DIR=/scratch/uv_cache -e UV_PYTHON_INSTALL_DIR=/scratch/uv_python \
  -e HF_HOME=/scratch/hf_cache -e HUGGINGFACE_HUB_CACHE=/scratch/hf_cache \
  -e FLASHINFER_WORKSPACE_DIR=/scratch/flashinfer_cache \
  -e TRITON_CACHE_DIR=/scratch/triton_cache \
  -e XDG_CACHE_HOME=/scratch/cache \
  novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c \"ray start --head --port=6379 --dashboard-host=0.0.0.0 && sleep infinity\"
EOF"

echo "=== Step 4: Start worker node ==="
srun --overlap --jobid=$JOBID --nodes=1 --nodelist=$NODE2 bash -c "DOCKER_GID=\$(getent group docker | cut -d: -f3) && newgrp docker << EOF
docker rm -f skyrl-worker 2>/dev/null
docker run -d --runtime=nvidia --gpus all --cpuset-cpus=0-167 --shm-size=16g \
  --group-add \$DOCKER_GID \
  --network=host --name skyrl-worker \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v /home/hkang/wl/SkyRL:/workspace/SkyRL \
  -v $SCRATCH_DIR:/scratch \
  -e RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
  -e DATA=/scratch -e HOME=/scratch \
  -e UV_CACHE_DIR=/scratch/uv_cache -e UV_PYTHON_INSTALL_DIR=/scratch/uv_python \
  -e HF_HOME=/scratch/hf_cache -e HUGGINGFACE_HUB_CACHE=/scratch/hf_cache \
  -e FLASHINFER_WORKSPACE_DIR=/scratch/flashinfer_cache \
  -e TRITON_CACHE_DIR=/scratch/triton_cache \
  -e XDG_CACHE_HOME=/scratch/cache \
  novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c \"ray start --address=$NODE1:6379 && sleep infinity\"
EOF"

sleep 5
echo "=== Step 5: Verify cluster ==="
srun --overlap --jobid=$JOBID --nodes=1 --nodelist=$NODE1 bash -c 'newgrp docker << EOF
docker exec skyrl-head ray status
EOF'

echo "=== Step 6: Copy code to scratch ==="
srun --overlap --jobid=$JOBID --nodes=2 --ntasks=2 bash -c 'newgrp docker << \EOF
CONTAINER_ID=$(docker ps -q -f "name=skyrl-")
if [ -n "$CONTAINER_ID" ]; then
    echo "Syncing code on $(hostname) in container $CONTAINER_ID..."
    docker exec $CONTAINER_ID bash -c "rm -rf /scratch/SkyRL && cp -r /workspace/SkyRL /scratch/SkyRL"
fi
EOF'

echo "=== Step 7: Prepare data ==="
srun --overlap --jobid=$JOBID --nodes=1 --nodelist=$NODE1 bash -c 'newgrp docker << EOF
docker exec skyrl-head bash -c "export HF_HOME=/scratch/hf_cache && cd /scratch/SkyRL/skyrl-train && uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir /scratch/swe_gym_subset"
EOF'

echo "Setup done."
