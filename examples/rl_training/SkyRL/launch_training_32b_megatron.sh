#!/bin/bash
# launch_training_32b_megatron.sh - Runs inside the Docker container on the HEAD node
# Starts ThunderAgent proxy + external profiler as background processes, then runs 32B Megatron fully async training
# Usage: docker exec skyrl-head bash /scratch/SkyRL/launch_training_32b_megatron.sh [extra_hydra_overrides...]
#
# Environment variables:
#   THUNDERAGENT_ROUTER  - "default" (pure proxy) or "tr" (capacity scheduling). Default: "default"
#   ACTING_TOKEN_WEIGHT  - Token weight for TR router. Default: 1.0
#   HEAD_NODE            - Hostname of the head node (for OPENAI_BASE_URL). Default: $(hostname)
#   SLURM_JOB_ID        - SLURM job ID (for per-job profiler dirs)
#   NUM_INFERENCE_ENGINES - Number of vLLM engines (for per-engine routing). Default: 4
#
# Architecture (6-node, per-engine routing):
#   LiteLLM (port 8001) → ThunderAgent (port 8001) → SkyRL /engines/{id}/... (port 8002) → N vLLM engines (TP=2)

set -e

ROUTER_MODE="${THUNDERAGENT_ROUTER:-default}"
HEAD_NODE="${HEAD_NODE:-$(hostname)}"
echo "=== ThunderAgent router mode: $ROUTER_MODE ==="
echo "=== HEAD_NODE: $HEAD_NODE ==="

# Source environment
source /workspace/SkyRL/api-keys.sh
source /workspace/SkyRL/env_vars.sh
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HEAD_NODE
# Reduce Ray GCS task event overhead (defense-in-depth; must be set before ray start
# on each node to affect workers — already in sbatch script's docker -e flags)
export RAY_task_events_report_worker_events=false
# H100 arch list — needed for megatron-core import on CPU-only HEAD node
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
# UV cache: inherit from container env (HEAD uses /uv_cache_ram tmpfs, workers use /scratch/uv_cache_v2)
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/uv_cache_v2}"
mkdir -p "$UV_CACHE_DIR" 2>/dev/null

echo "=== Environment configured ==="

# Install ThunderAgent if not already installed
if ! python3 -c "import ThunderAgent" 2>/dev/null; then
    echo "=== Installing ThunderAgent ==="
    pip install -e /scratch/ThunderAgent-wl 2>&1 | tail -5
fi

# Per-job profiler directories (avoid data contamination when nodes are reused)
PROFILER_DIR="/scratch/profiler_data/job_${SLURM_JOB_ID:-unknown}"
THUNDER_PROFILE_DIR="/scratch/thunderagent_profiles/job_${SLURM_JOB_ID:-unknown}"

# Launch external profiler as background process
mkdir -p "$PROFILER_DIR"
cd /scratch/SkyRL
nohup python3 external_profiler.py \
  --output-dir "$PROFILER_DIR" \
  --poll-interval 5 \
  --http-url http://127.0.0.1:8002/metrics \
  --docker-interval 30 \
  > "$PROFILER_DIR/profiler.log" 2>&1 &
PROFILER_PID=$!
echo "=== External profiler started (PID=$PROFILER_PID), dir=$PROFILER_DIR ==="

NUM_ENGINES="${NUM_INFERENCE_ENGINES:-4}"
TRAIN_LOG="/tmp/training_inner_$$.log"

# Launch training in background first (SkyRL HTTP endpoint starts on port 8002)
# The entrypoint is a Ray task that may run on ANY node in the cluster.
cd /scratch/SkyRL/skyrl-train
echo "=== Starting 32B Megatron fully async training with args: $@ ==="
bash examples/mini_swe_agent/run_mini_swe_32B_megatron_fully_async.sh "$@" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "=== Training launched in background (PID=$TRAIN_PID) ==="

# Tail training log to stdout in background so it appears in the outer log
tail -f "$TRAIN_LOG" --pid=$TRAIN_PID &
TAIL_PID=$!

# Detect the entrypoint IP by watching for HTTP endpoint startup message.
# The Ray task that hosts port 8002 can be on any node (not necessarily HEAD).
echo "=== Waiting for HTTP endpoint to start (detecting entrypoint IP)... ==="
ENTRYPOINT_IP=""
for i in $(seq 1 600); do
  if [ -f "$TRAIN_LOG" ]; then
    # Look for the entrypoint IP from the Ray task prefix: (skyrl_entrypoint pid=NNN, ip=X.X.X.X)
    # Strip ANSI color codes first (Ray logs contain [36m etc.)
    ENTRYPOINT_IP=$(sed 's/\x1b\[[0-9;]*m//g' "$TRAIN_LOG" 2>/dev/null | grep -oP 'skyrl_entrypoint pid=\d+, ip=\K[0-9.]+' | head -1)
    if [ -n "$ENTRYPOINT_IP" ]; then
      # Verify the HTTP endpoint is actually ready
      if curl -sf "http://${ENTRYPOINT_IP}:8002/v1/models" > /dev/null 2>&1; then
        echo "=== Entrypoint detected at ${ENTRYPOINT_IP}:8002 (after ${i}s) ==="
        break
      fi
    fi
  fi
  sleep 1
done

if [ -z "$ENTRYPOINT_IP" ]; then
  echo "WARNING: Could not detect entrypoint IP after 600s, falling back to 127.0.0.1"
  ENTRYPOINT_IP="127.0.0.1"
fi

# Build per-engine backend URLs pointing to the actual entrypoint node.
BACKENDS=""
for i in $(seq 0 $((NUM_ENGINES - 1))); do
  [ -n "$BACKENDS" ] && BACKENDS="${BACKENDS},"
  BACKENDS="${BACKENDS}http://${ENTRYPOINT_IP}:8002/engines/${i}"
done
echo "=== ThunderAgent backends ($NUM_ENGINES engines): $BACKENDS ==="

# Start ThunderAgent on port 8001 (where LiteLLM connects via OPENAI_BASE_URL)
mkdir -p "$THUNDER_PROFILE_DIR"
echo "=== ThunderAgent: starting with --router $ROUTER_MODE, backends on $ENTRYPOINT_IP ==="
python3 -m ThunderAgent \
  --host 0.0.0.0 \
  --port 8001 \
  --backends "$BACKENDS" \
  --router "$ROUTER_MODE" \
  --backend-type skyrl \
  --profile \
  --profile-dir "$THUNDER_PROFILE_DIR" \
  --metrics \
  --metrics-interval 5.0 \
  --scheduler-interval 5.0 \
  --acting-token-weight ${ACTING_TOKEN_WEIGHT:-1.0} \
  --log-level info &
THUNDERAGENT_PID=$!
echo "=== ThunderAgent started (PID=$THUNDERAGENT_PID) on port 8001 ==="

# Update external profiler to poll the entrypoint's metrics endpoint
kill $PROFILER_PID 2>/dev/null || true
cd /scratch/SkyRL
nohup python3 external_profiler.py \
  --output-dir "$PROFILER_DIR" \
  --poll-interval 5 \
  --http-url "http://${ENTRYPOINT_IP}:8002/metrics" \
  --docker-interval 30 \
  > "$PROFILER_DIR/profiler.log" 2>&1 &
PROFILER_PID=$!
echo "=== External profiler restarted with entrypoint IP $ENTRYPOINT_IP (PID=$PROFILER_PID) ==="

# Wait for training to finish
wait $TRAIN_PID
TRAIN_EXIT=$?

# Stop ThunderAgent and profiler when training finishes
kill $THUNDERAGENT_PID 2>/dev/null || true
echo "=== ThunderAgent stopped ==="
kill $PROFILER_PID 2>/dev/null || true
echo "=== Profiler stopped ==="
kill $TAIL_PID 2>/dev/null || true

exit $TRAIN_EXIT
