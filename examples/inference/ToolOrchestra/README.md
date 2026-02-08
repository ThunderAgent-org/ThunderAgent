# Overview

This folder contains reproducible end-to-end guides for running HLE (Humanity's Last Exam) evaluations with **ThunderAgent + vLLM** using the ToolOrchestra multi-agent workflow.

![ThunderAgent overview](../docs/thunder/figures/thunder.jpg)

## ToolOrchestra + ThunderAgent Use Case

### Prerequisites
- Python 3.12
- [Conda](https://docs.conda.io/en/latest/) package manager
- 2× GPUs (tested on RTX 5090) with CUDA drivers
- API keys: OpenAI (`OPENAI_API_KEY`), Together AI (`TOGETHER_API_KEY`), Tavily (`TAVILY_KEY`)

### Intro

Run the HLE benchmark through ToolOrchestra's multi-agent workflow, where:
- **Orchestrator-8B** (local, served via vLLM) coordinates tool calls
- **Expert Models** (external APIs) handle specific tasks:
  - `search` → OpenAI `gpt-5-mini`
  - `enhance_reasoning` → OpenAI `gpt-5`
  - `answer` → Together AI `openai/gpt-oss-120b`

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HLE Evaluation                              │
├─────────────────────────────────────────────────────────────────────┤
│  eval_hle_local.py                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────┐                                               │
│  │ ThunderAgent     │◄──── Manages KV cache, scheduling             │
│  │ Router           │      (--router tr or --router default)        │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐         ┌─────────────────────────────────┐   │
│  │ vLLM Server      │         │ Expert APIs (External)          │   │
│  │ Orchestrator-8B  │         │ • gpt-5-mini (search)           │   │
│  │ (GPU 0)          │         │ • gpt-5 (enhance_reasoning)     │   │
│  └──────────────────┘         │ • gpt-oss-120b (answer)         │   │
│                               └─────────────────────────────────┘   │
│  ┌──────────────────┐                                               │
│  │ Retriever        │◄──── FAISS GPU index for document search      │
│  │ (GPU 1)          │                                               │
│  └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Environment Setup

We use three conda environments:

```bash
# vLLM environment (for Orchestrator-8B serving)
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.10.1 transformers hf_transfer matplotlib

# vLLM-Continuum environment (for Continuum scheduler comparison)
# Follow instructions in vllm-continuum repo

# Retriever environment (for FAISS GPU index)
conda create -n retriever-clean python=3.12 -y
conda activate retriever-clean
conda install -y -c conda-forge --force-reinstall "numpy<2" "scipy<2" "scikit-learn<2" numpy-base
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
conda install -y -c pytorch -c nvidia faiss-gpu
pip install packaging ninja psutil
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install transformers datasets uvicorn fastapi tavily-python hf_transfer
```

### 2. Verify Environments

```bash
# Verify retriever environment
conda activate retriever-clean
python -c 'import faiss; n=faiss.get_num_gpus(); print(f"Faiss GPUs: {n}"); assert n > 0'

# Verify vLLM environments
conda activate vllm1
python -c "from vllm import LLM; print('vllm1 OK')"
```

### 3. Download Required Data

```bash
# Clone the retrieval index
git clone https://huggingface.co/datasets/multi-train/index
export INDEX_DIR='/path/to/index'

# Clone the Orchestrator model
git clone https://huggingface.co/nvidia/Nemotron-Orchestrator-8B
export CKPT_DIR='/path/to/Nemotron-Orchestrator-8B'
```

### 4. Configure Environment Variables

```bash
# Copy the template and fill in your values
cp setup_envs.sh_example setup_envs.sh

# Edit setup_envs.sh with your paths and API keys:
#   CKPT_DIR, INDEX_DIR, THUNDERAGENT_ROOT, OPENAI_API_KEY, TOGETHER_API_KEY, TAVILY_KEY

# Source the environment
source setup_envs.sh
```

## Figure Reproduction
These scripts reproduce the throughput plot in the paper for **ToolOrchestra (HLE) – Qwen3-8B**.

Test hardware: 2× RTX 5090 (Orchestrator on GPU0, Retriever on GPU1) + external expert APIs.

Scripts:
- `scripts/reproduce/reproduce_hle_qwen3_8b.sh` runs C ∈ {24, 32, 40, 48} for `baseline`, `continuum`, `thunderagent`.
- `scripts/reproduce/plot_hle_qwen3_8b.py` reads the latest runs and draws the plot.

```bash
# Run all settings (this is long: 12 runs at 2h30 each by default)
./scripts/reproduce/reproduce_hle_qwen3_8b.sh

# Plot throughput (step/min)
./scripts/reproduce/plot_hle_qwen3_8b.py
# Output: evaluation/reports/hle_qwen3_8b_throughput.png
```

Expected result (throughput comparison):
![throughput_compare](../docs/toolorchestra/throughput_compare.png)


## How to Run Experiments

### Quick Start

```bash
# Source environment variables first
source setup_envs.sh

# Run one setting with defaults:
#   METHOD=thunderagent, CONCURRENCY=48, EVAL_TIMEOUT_MIN=150
#   WINDOW_START_SEC=600, WINDOW_END_SEC=7800
./run_single_setting.sh
```

Override defaults (example):
```bash
export METHOD=baseline
export CONCURRENCY=64
export EVAL_TIMEOUT_MIN=180
export WINDOW_START_SEC=900
export WINDOW_END_SEC=9000
./run_single_setting.sh
```

What this does:
- Starts **retriever**, **vLLM backend**, and **ThunderAgent router** in one shot.
- Enables prefix caching + prompt token details + usage logging on vLLM.
- Starts metrics samplers for `/metrics` and GPU SM utilization.
- Runs `eval_hle_local.py` with your `--concurrency`.
- Writes output to `evaluation/outputs/hle_local_YYYYMMDD_HHMMSS/`.
- Writes logs to `evaluation/logs/hle_local_YYYYMMDD_HHMMSS/`.
- Produces: `orchestrator_usage.jsonl`, `prefix_cache_timeseries.csv`,
  `gpu_sm_util_timeseries.csv`, `window_summary.json`, `steps_summary.json`,
  `combined_summary.json`.

To inspect the latest run:
```bash
LATEST_OUT=$(ls -td evaluation/outputs/hle_local_* | head -1)
LATEST_LOG=$(ls -td evaluation/logs/hle_local_* | head -1)
echo "$LATEST_OUT"
echo "$LATEST_LOG"
```

To view throughput + KV cache hit rates for the active window:
```bash
cat "$LATEST_OUT/combined_summary.json"
```

To print key metrics from the latest run:
```bash
python - <<'PY'
import json, os, subprocess

latest_out = subprocess.check_output(
    "ls -td evaluation/outputs/hle_local_* | head -1",
    shell=True,
    text=True,
).strip()

path = os.path.join(latest_out, "combined_summary.json")
data = json.load(open(path, "r", encoding="utf-8"))

window = data.get("windows", {}) or {}
win_key = next(iter(window.keys()), None)
metrics = (window.get(win_key, {}) or {}).get("metrics", {}) or {}
usage = (window.get(win_key, {}) or {}).get("response_usage", {}) or {}
steps = (data.get("steps") or {}).get("steps_per_sec")

print("out_dir:", latest_out)
print("window:", win_key)
print("trials/min:", (window.get(win_key, {}) or {}).get("trials", {}).get("trials_per_sec", 0) * 60 if win_key else "n/a")
print("steps/sec:", steps)
print("server_hit_ratio:", metrics.get("hit_ratio"))
print("request_hit_avg:", usage.get("cached_tokens_ratio_request_avg"))
print("request_hit_token_weighted:", usage.get("cached_tokens_ratio_token_weighted"))
print("kv_usage_mean_perc:", metrics.get("kv_cache_usage_mean_perc"))
PY
```

### Launch Script Options

```
Usage:
  bash evaluation/launch_hle_inference.sh --method <METHOD> --concurrency <C> \
    [--ckpt <CKPT_DIR>] [--index-dir <INDEX_DIR>] [options]

Required:
  --method         baseline | continuum | thunderagent
  --concurrency    HLE concurrency (eval batch size)
  --ckpt           Orchestrator-8B checkpoint path (or set CKPT_DIR env)
  --index-dir      Directory containing eval.index + eval.jsonl (or set INDEX_DIR env)

Common options:
  --orchestrator-gpu  GPU for vLLM (default: 0)
  --retrieval-gpu     GPU for retriever (default: 1)
  --backend-port      vLLM backend port (default: 8100)
  --router-port       ThunderAgent router port (default: 8000)
  --max-rounds        Max rounds per task (default: 50)
  --log-level         HLE log level (default: INFO)
  --eval-timeout-min  Timeout minutes for eval (default: 0 = no timeout)
  --window-start-sec  Active window start offset in seconds (default: 600)
  --window-end-sec    Active window end offset in seconds (default: 7800)
  --sample-interval-sec  Metrics sampling interval seconds (default: 2)
  --thunderagent-root Path to ThunderAgent repo (default: /workspace/ThunderAgent or $THUNDERAGENT_ROOT; usually set explicitly)

Outputs (when using `evaluation/launch_hle_inference.sh`):
- `prefix_cache_timeseries.csv`, `gpu_sm_util_timeseries.csv`
- `window_summary.json`, `steps_summary.json`, `combined_summary.json`
- `orchestrator_usage.jsonl` (used by window summaries)
```

## What We Changed for ThunderAgent Integration

To integrate ThunderAgent with your own agent workflow, you need two key changes:

### 1. Program ID Injection

Location: [`evaluation/eval_hle_local.py`](evaluation/eval_hle_local.py) (`run_single()`).

We give each HLE trial its own `program_id`, then attach it to every orchestrator call via `extra_body.program_id`. You can think of `program_id` as the thread label ThunderAgent uses to keep KV cache and pause/resume state isolated per task. Without it, trials can get mixed together.

```python
# Create a unique program_id per HLE trial
import uuid
program_id = f"hle:{e['id']}:{uuid.uuid4().hex[:8]}"

# Pass program_id in extra_body for every orchestrator request
response = get_llm_response(
    model=MODEL_NAME,
    messages=chat,
    extra_body={"program_id": program_id},
)
```

### 2. Program Release Hook

Location: [`evaluation/eval_hle_local.py`](evaluation/eval_hle_local.py) (end of `run_single()` via `_release_program()`).

When a trial ends, we send `POST /programs/release` with that same `program_id`. This tells the router “you can forget this task now,” clearing KV/cache bookkeeping so finished trials don’t keep consuming capacity.

```python
# After trial completes, release the program_id
import requests

router_url = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
try:
    requests.post(
        f"{router_url}/programs/release",
        json={"program_id": program_id},
        timeout=5
    )
except Exception:
    pass  # Best-effort cleanup
```

## Methods Comparison

| Method | vLLM Env | Router Mode | Scheduler |
|--------|----------|-------------|-----------|
| `baseline` | vllm1 | `--router default` | Standard vLLM FCFS |
| `continuum` | vllm-continuum | `--router default` | Continuum TTL pinning |
| `thunderagent` | vllm1 | `--router tr` | ThunderAgent capacity scheduling |

## Expert Model Configuration

| Tool | Expert Model | Provider | API Key |
|------|--------------|----------|---------|
| `search` | gpt-5-mini | OpenAI | `OPENAI_API_KEY` |
| `enhance_reasoning` | gpt-5 | OpenAI | `OPENAI_API_KEY` |
| `answer` | openai/gpt-oss-120b | Together AI | `TOGETHER_API_KEY` |

## Repository Layout

```text
ToolOrchestra/
├── README.md                    # This file
├── setup_envs.sh_example        # Environment template (copy to setup_envs.sh)
├── LLM_CALL.py                  # Unified LLM interface (OpenAI, Together, vLLM)
├── evaluation/
│   ├── launch_hle_inference.sh  # Main launch script
│   ├── eval_hle_local.py        # HLE evaluation with local Orchestrator
│   ├── retrieval_hle.py         # FAISS GPU retrieval service
│   ├── hle.jsonl                # HLE benchmark dataset
│   ├── tools.json               # Tool definitions
│   ├── model_configs/           # Generated model config files
│   ├── outputs/                 # Evaluation outputs
│   └── logs/                    # Service logs
├── scripts/
│   └── reproduce/               # Paper reproduction scripts
```

## Troubleshooting

### Common Issues

1. **vLLM server not starting**: Check GPU memory and CUDA drivers. Orchestrator-8B requires ~16GB VRAM.

2. **Retriever FAISS errors**: Ensure `eval.index` and `eval.jsonl` exist in `INDEX_DIR`.

3. **API key errors**: Verify `OPENAI_API_KEY` and `TOGETHER_API_KEY` are set correctly.

4. **Module not found**: Ensure `PYTHONPATH` includes `/workspace/ThunderAgent`.

### Logs

Check logs in `evaluation/logs/hle_local/`:
- `vllm_backend.log` - vLLM server logs
- `router.log` - ThunderAgent router logs
- `retrieval.log` - Retriever service logs
