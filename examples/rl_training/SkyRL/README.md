# SkyRL Use Case: Post-Training Qwen3-14B in Mini-SWE-Agent Environment

## Overview
This directory provides a concrete use case of using the SkyRL training framework to post-train `Qwen/Qwen3-14B` in the Mini-SWE-Agent environment on a SLURM cluster.

Within this use case, we run and compare rollout throughput (**tokens/sec**) for:

- `dp4_default`: ThunderAgent router `default` (passthrough behavior, equivalent to standard vLLM request routing, with additional ThunderAgent profiling/metrics modules enabled)
- `dp4_tr_atw01`: program-aware ThunderAgent router `tr` with `acting_token_weight=0.1` that automatically pauses, resumes and transfer programs across backends

This workflow is intended as a reference example under:

- `<THUNDERAGENT_ROOT>/examples/rl_training/`

## Reproduction Target
The goal is to reproduce the **per-step rollout throughput (tokens/sec)** comparison under:

- Model: `Qwen/Qwen3-14B`
- Inference layout: DP=4, TP=1 (4 vLLM engines)
- Training layout: FSDP2, SP=4, non-colocated (4 training GPUs)
- Data: `SumanthRH/SWE-Gym-Subset`
- Batch config: `train_batch_size=99`, `n_samples_per_prompt=4` (396 trajectories/step)

Throughput is computed per rollout step as:

- `tokens/sec = (num_trajectories_per_step * avg_response_length) / generate_duration_seconds`
- In this setup, `num_trajectories_per_step = 396` (99 prompts x 4 samples per prompt)

## Expected Directory Layout
This README assumes a layout like:

```text
<THUNDERAGENT_ROOT>/
├── examples/
│   └── rl_training/
│       └── SkyRL/                      # this repo
└── ThunderAgent/                       # python package source
```

In this layout:

- `SKYRL_HOST_DIR=<THUNDERAGENT_ROOT>/examples/rl_training/SkyRL`
- `THUNDERAGENT_HOST_DIR=<THUNDERAGENT_ROOT>`

If your layout differs, set these paths explicitly (see "Required Customization").

## Prerequisites
- SLURM cluster access (1 node, 8 GPUs, tested with H100 80GB)
- Docker available on compute nodes
- Permission to run `sudo` for Docker daemon configuration (`/etc/docker/daemon.json`)
- Python 3.10+ on the submission host
- WANDB API key (required)
- HuggingFace token (recommended for model/dataset/image pulls)

## Required Customization
Before running, customize the following.

1. `api-keys.sh` in SkyRL root:
```bash
export WANDB_API_KEY="<your_wandb_key>"
# optional but recommended
export HF_TOKEN="<your_hf_token>"
```

2. SBATCH header fields in:
- `sbatch_repro_dp4_default.sh`
- `sbatch_repro_dp4_tr_atw01.sh`

Typical fields to edit for your cluster:
- `#SBATCH --partition=...`
- `#SBATCH --time=...`
- `#SBATCH --cpus-per-task=...`
- `#SBATCH --gpus=...`
- optional: `#SBATCH --account=...`, `#SBATCH --qos=...`, `#SBATCH --exclude=...`

3. Runtime path/image variables (export before submission):
```bash
export SKYRL_HOST_DIR="<path-to-SkyRL>"                           # host path mounted to /workspace/SkyRL in the training container
export THUNDERAGENT_HOST_DIR="<path-to-ThunderAgent-wl-root>"     # host path mounted to /workspace/ThunderAgent-wl
export TRAINING_DATA_SYNC_DEST="<where-to-sync-training-artifacts>" # destination directory for periodic rsync from /scratch
export DOCKER_IMAGE="novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8" # training image used by sbatch reproducibility scripts
```

If not set, defaults are inferred from the expected layout above.

## Setup
From your SkyRL directory:

```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

# Optional: verify key files
ls launch_profiled_training_14b_dp4.sh
ls skyrl-train/examples/mini_swe_agent/run_mini_swe_14B_dp4.sh
```

Prepare credentials:

```bash
cat > api-keys.sh <<'SH'
export WANDB_API_KEY="<your_wandb_key>"
export HF_TOKEN="<your_hf_token>"
SH
chmod 600 api-keys.sh
```

## Reproduction
### Option A: Submit both jobs with one command
```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

export SKYRL_HOST_DIR="$(pwd)"
export THUNDERAGENT_HOST_DIR="$(cd ../../.. && pwd)"
export TRAINING_DATA_SYNC_DEST="$(pwd)/training_data"

bash submit_repro_dp4_default_vs_tr_atw01.sh
```

### Option B: Submit jobs separately
```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

sbatch --export=ALL sbatch_repro_dp4_default.sh
sbatch --export=ALL sbatch_repro_dp4_tr_atw01.sh
```

## Monitoring
Check job states:

```bash
squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.20N"
```

Check logs:

```bash
tail -n 40 slurm-<JOBID>.out
tail -n 60 slurm-<JOBID>.err
```

Useful rollout markers:

```bash
grep "Started: 'step'" slurm-<JOBID>.err
grep "Finished: 'generate'" slurm-<JOBID>.err
grep "avg_response_length" slurm-<JOBID>.err
```

## Throughput Analysis (tokens/sec only)
After both jobs complete at least one rollout step:

```bash
python3 analyze_dp4_tokens_compare.py \
  --default-job <DEFAULT_JOB_ID> \
  --tr-job <TR_JOB_ID>
```

The script prints a focused markdown table:

- one row per rollout step
- `dp4_default tokens/sec`
- `dp4_tr_atw01 tokens/sec`
- delta and delta percentage
- average over paired completed steps

Expected output:

```text
# DP=4 Rollout Tokens/sec Comparison (dp4_default vs dp4_tr_atw01)

- default job: `25609`
- tr job: `25610`
- formula: `tokens/sec = (396 * avg_response_length) / generate_duration_s`

Step                   | dp4_default tokens/sec | dp4_tr_atw01 tokens/sec | Delta (tr-default) | Delta %
-----------------------+------------------------+-------------------------+--------------------+--------
1                      |                 1222.7 |                  1385.9 |              163.2 |    13.3
2                      |                 1362.8 |                  1713.0 |              350.2 |    25.7
3                      |                 1776.8 |                  1809.6 |               32.8 |     1.8
4                      |                 1645.3 |                  1943.9 |              298.6 |    18.1
5                      |                 1747.6 |                  1917.3 |              169.7 |     9.7
6                      |                 1207.6 |                  1842.4 |              634.8 |    52.6
Average (paired steps) |                 1493.8 |                  1768.7 |              274.9 |    18.4
```

## Reproduction Scripts in This Directory
- `sbatch_repro_dp4_default.sh`: explicit DP=4 default-router job
- `sbatch_repro_dp4_tr_atw01.sh`: explicit DP=4 TR job with `acting_token_weight=0.1`
- `scripts/repro/run_dp4_repro_job.sh`: shared SLURM job body used by both scripts
- `submit_repro_dp4_default_vs_tr_atw01.sh`: submit both jobs and print job IDs
- `analyze_dp4_tokens_compare.py`: focused per-step tokens/sec comparison table

## Notes
- Training artifacts are synced to `TRAINING_DATA_SYNC_DEST/job_<JOBID>_dp4_<tag>/`.
