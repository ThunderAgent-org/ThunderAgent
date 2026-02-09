#!/bin/bash
# run_training.sh - Runs the training command
# Usage: bash run_training.sh [JOBID] [NODE]

export JOBID=${1:-25363}
export NODE1=${2:-research-secure-10}

echo "Starting training on JOBID=$JOBID, NODE=$NODE1..."
echo "Using scratch dir: /scratch/$USER (mounted as /scratch in container)"

# Sync code on all nodes (head and workers) before training
echo "Syncing code to all nodes..."
srun --overlap --jobid=$JOBID --ntasks-per-node=1 bash -c 'newgrp docker << \EOF
CONTAINER_ID=$(docker ps -q -f "name=skyrl-")
if [ -n "$CONTAINER_ID" ]; then
    echo "Syncing code on $(hostname) in container $CONTAINER_ID..."
    docker exec $CONTAINER_ID bash -c "rm -rf /scratch/SkyRL && cp -r /workspace/SkyRL /scratch/SkyRL"
fi
EOF'

echo "Cleaning up old minisweagent containers on all nodes..."
srun --overlap --jobid=$JOBID --ntasks-per-node=1 bash -c 'newgrp docker << \EOF
echo "Cleaning minisweagent containers on $(hostname)..."
docker ps -a --filter "name=minisweagent" -q | xargs -r docker rm -f 2>/dev/null || true
echo "Done cleaning on $(hostname)"
EOF'

echo "Launching training..."
SYNC_DEST="/home/$USER/wl/SkyRL/training_data"
mkdir -p "$SYNC_DEST"
srun --overlap --oversubscribe --jobid=$JOBID --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --nodelist=$NODE1 bash -c '
# Background sync: copy training data from local scratch to NFS every 60s
mkdir -p '"$SYNC_DEST"'
(while true; do
    sleep 60
    for d in /scratch/'"$USER"'/mini_swe_agent_trajs* /scratch/'"$USER"'/profiler_data; do
        [ -d "$d" ] && rsync -a --ignore-errors "$d" '"$SYNC_DEST"'/ 2>/dev/null
    done
done) &
SYNC_PID=$!
echo "Background sync started (PID=$SYNC_PID): /scratch/'"$USER"' -> '"$SYNC_DEST"' every 60s"

newgrp docker << EOF
docker exec skyrl-head bash -c "source /workspace/SkyRL/api-keys.sh && source /workspace/SkyRL/env_vars.sh && export HYDRA_FULL_ERROR=1 && cd /scratch/SkyRL/skyrl-train && bash examples/mini_swe_agent/run_mini_swe_30B.sh trainer.eval_before_train=false trainer.eval_interval=0 trainer.policy.sequence_parallel_size=1"
EOF

kill $SYNC_PID 2>/dev/null
echo "Background sync stopped"
'
