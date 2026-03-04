# Reproducing 32B Qwen3 Fully Async GRPO Training on SWE-Gym

## Overview

This guide walks through reproducing 32B Qwen3 fully async GRPO training on SWE-Gym,
comparing ThunderAgent **default** vs **TR** router modes.

The experiment uses 6 nodes (48 GPUs allocated, 40 used) with Megatron-based training
(TP=4, PP=4, CP=2, DP=1) and 4 vLLM TP=2 inference engines.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ NODE 0 (HEAD): Ray GCS + 128 SWE workers + Docker-in-Docker      │
│   - ray start --num-gpus=0 (8 GPUs idle, allocated by SLURM)     │
│   - ThunderAgent proxy (port 8001) + SkyRL entrypoint (8002)     │
│                                                                  │
│ NODE 1 (Inference): 4 vLLM engines × TP=2 (DP=4)         8 GPUs  │
│                                                                  │
│ NODES 2-5 (Training): Megatron TP=4 PP=4 CP=2 DP=1      32 GPUs  │
│   - PP stages 0-3, one stage per node                            │
└──────────────────────────────────────────────────────────────────┘
Total: 48 GPUs allocated by SLURM, 40 used (8 idle on HEAD node)
```

**Request flow**:
```
SWE Worker → LiteLLM (port 8001) → ThunderAgent (port 8001)
  → SkyRL /engines/{id}/... (port 8002) → vLLM engine (TP=2)
```

## Prerequisites

- **6 nodes** with 8 GPUs each (tested on H100 80GB)
- **Docker** available on all nodes with `sudo` access for daemon configuration
- **SLURM** with multi-node allocation support (`--nodes=6`)
- **~65 GB** disk per node for Qwen3-32B model weights
- **~500 GB** disk on HEAD node for SWE-bench Docker images (2400+)
- **WandB API key** (required for training metrics)
- **HuggingFace token** (required for model/dataset download)

## Quick Start

### 1. Set up credentials

```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

# Copy and fill in your credentials
cp api-keys.sh.example api-keys.sh
chmod 600 api-keys.sh
# Edit api-keys.sh with your WANDB_API_KEY and HF_TOKEN
```

### 2. Customize cluster config (optional)

```bash
cp cluster-config.sh.example cluster-config.sh
# Edit cluster-config.sh:
#   - NCCL_SOCKET_IFNAME: your inter-node network interface (find with: ip route | grep default | awk '{print $5}')
#   - NCCL_IB_DISABLE: set to 1 if InfiniBand is unavailable inside Docker
```

### 3. Customize SBATCH headers

Edit `sbatch_32b_megatron_default.sh` and `sbatch_32b_megatron_tr.sh`:
- `--partition`: your cluster's partition name
- `--cpus-per-task`: number of CPUs per node
- `--time`: wall time limit
- Optional: `--account`, `--qos`, `--exclude`

### 4. Submit both experiments

```bash
bash submit_32b_megatron_default_vs_tr.sh
```

Or submit individually:
```bash
sbatch --export=ALL sbatch_32b_megatron_default.sh
sbatch --export=ALL sbatch_32b_megatron_tr.sh
```

## Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-32B | 17 safetensor shards, ~65 GB |
| Algorithm | GRPO | `trainer.algorithm.advantage_estimator=grpo` |
| Strategy | Megatron | TP=4, PP=4, CP=2, DP=1 |
| vLLM engines | 4 × TP=2 | `gpu_memory_utilization=0.85` |
| train_batch_size | 128 | prompts per step |
| n_samples | 4 | samples per prompt → 512 trajectories/step |
| max_training_seq_len | 40960 | ~40K tokens max |
| max_input_length | 40960 | matches training seq len |
| max_turns | 20 | SWE agent turns per trajectory |
| parallel workers | 128 | concurrent SWE environments |
| gradient_checkpointing | true | required for 32B |
| micro_batch_size | 1 | per-device |
| gradient_accumulation | 4 | effective batch = 4 × DP(1) = 4 micro-batches |

## Monitoring

### WandB Metrics
Key metrics to watch:
- `policy_loss`: should decrease over steps
- `pass_at_1`, `pass_at_4`: solve rates
- `num_real_trajectories` / `num_dummy_trajectories`: dummy ratio should stay <2%
- `staleness`: weight version lag between inference and training

### SLURM Logs
```bash
# Check job status
squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.20N"

# Training progress
grep "Started: 'step'" slurm-<JOBID>.err
grep "Finished: 'generate'" slurm-<JOBID>.err

# Infrastructure metrics
tail -f slurm-<JOBID>.err
```

### Profiler Data
Infrastructure metrics are collected every 5s by `external_profiler.py`:
- Output: `/scratch/profiler_data/job_<JOBID>/`
- Synced to: `training_data/job_<JOBID>_32b_megatron_<tag>/`

## Known Issues & Workarounds

### NCCL Cross-Node in Docker
Multi-node NCCL requires correct network interface configuration:
```bash
# In cluster-config.sh:
export NCCL_SOCKET_IFNAME="ens7"    # your inter-node interface
export NCCL_IB_DISABLE=1            # if no InfiniBand in Docker
```
Without this, NCCL picks Docker's bridge interface (not routable cross-node) and hangs.

### KV Cache Pressure
With 4 vLLM engines at `gpu_memory_utilization=0.85`, KV cache runs at ~91.7% utilization.
This is tight but stable. Reducing to 0.80 wastes inference throughput; increasing to 0.90
risks OOM during weight loading.

### Epoch Boundary Idle Time
At dataset epoch boundaries, the buffer drains and training idles for ~17 minutes while
new trajectories are generated. This is expected behavior for fully async training.

### Docker Hub Rate Limits
SWE-bench requires pulling 2400+ Docker images. Free-tier Docker Hub allows 200 pulls/6h.
Options:
1. Pre-cache images on your nodes
2. Use Docker Hub Pro
3. Configure multiple Docker Hub accounts in `api-keys.sh`

### PP Stage 0 Memory
Megatron PP stage 0 (embedding + first layers) runs hottest at ~78 GiB during `policy_train`
(96% of 80 GiB H100). The chunked logprob computation (`chunk_size=4096`) in
`megatron_model_wrapper.py` is critical — without it, computing logprobs over 40960-token
sequences would OOM instantly.

### TP=2/CP=4 vs TP=4/CP=2
TP=2 with CP=4 causes OOM. TP=4 with CP=2 fits (~27-35 GiB/GPU). Always use TP=4, CP=2
for 32B training.

<!-- ## Experimental Baseline (att29)

From 29 iterations of 32B Megatron fully async training:
- **43 training steps** completed over 27.3 hours
- **pass@1**: 2.03% average, **pass@4**: 4.84% average
- **Dummy rate**: <2% (most steps had 0 dummies)
- **Per-engine routing**: KV cache hit rate ~40%
- **Training speed**: ~40 min/step average (including generation + training)
- **OOM crash** at step 43 (PP stage 0 memory spike) — recoverable via checkpoint restart -->

<!-- ### Expected Results
- **pass@1**: ~2% avg, **pass@4**: ~5% avg
- **Training speed**: ~40 min/step
- **Dummy rate**: <2% (failed/timeout trajectories)
- **Steps completed**: 43 in 27.3 hours -->
