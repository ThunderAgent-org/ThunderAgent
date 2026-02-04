# mini-swe-agent + ThunderAgent Use Case

## Overview
This folder contains reproducible end-to-end guides for running SWE-bench evaluations with **ThunderAgent + vLLM** across **mini-swe-agent**.

![ThunderAgent overview](../docs/thunder/figures/thunder.jpg)

## Prerequisites
- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Docker daemon (SWE-bench instances run in Docker)
- GPU(s) with appropriate CUDA drivers


## Reproduction
These scripts reproduce the results reported in our paper.

Environment setup: [`setup.sh`](scripts/setup/setup.sh).

Test hardware: 8x H100 (NVIDIA driver 580.95.05 / CUDA 13.0).

- [`reproduce_glm4.6`](scripts/reproduce/reproduce_glm4.6): reproduce with GLM-4.6-FP8.
- [`reproduce_qwen3_235B`](scripts/reproduce/reproduce_qwen3_235B): reproduce with Qwen3-235B-A22B.

Expected result (throughput comparison):
![throughput_compare](../docs/mini-swe-agent/figures/throughput_compare.png)


## Setup
```bash
# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM (GPU build), mini-swe-agent in editable mode, and datasets
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/mini-swe-agent
uv pip install datasets
```

## How to run the experiment yourself

### One node example
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --port <VLLM_PORT>
```
2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://localhost:<VLLM_PORT> --port <TA_PORT>
```
3) Configure [`swebench.yaml`](src/minisweagent/config/benchmarks/swebench.yaml) to call ThunderAgent.


4) Run SWE-Bench via mini-swe-agent through ThunderAgent:
```bash
mini-extra swebench \
  --subset lite \
  --split test \
  --workers 128 \
  --output ./swebench_output 
```

### Multi nodes example
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --host 0.0.0.0 --port <VLLM_PORT>
```

```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --host 0.0.0.0 --port <VLLM_PORT>
```

2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://<VLLM_HOST1>:<VLLM_PORT>,http://<VLLM_HOST2>:<VLLM_PORT> --port <TA_PORT>
```
3) Configure [`swebench.yaml`](src/minisweagent/config/benchmarks/swebench.yaml) to call ThunderAgent.

2) Run SWE-Bench via mini-swe-agent through ThunderAgent:
```bash
mini-extra swebench \
  --subset lite \
  --split test \
  --workers 128 \
  --output ./swebench_output 
```


### What we changed in mini-swe-agent(to reuse in your own agent workflow)
- **Program ID injection**  
  Location: [`swebench.py`](src/minisweagent/run/benchmarks/swebench.py#L138) (`process_instance()`).  

  What: Assign a unique `program_id` per SWE-bench instance and pass it via `extra_body.program_id` on every model call.  

  Why: ThunderAgent uses this field to separate requests into per-program state (see [`get_program_id()` in app.py](../../../ThunderAgent/app.py#L42)).

  Implementation snippet:
  ```python
  # Create a unique program_id per SWE-bench instance and attach it to every request via extra_body.
  program_id = f"swe-{next(_PROGRAM_COUNTER):06d}"
  # Copy model config to avoid cross-thread mutation when running multiple instances in parallel.
  model_config = copy.deepcopy(config.get("model", {}))
  # ThunderAgent reads program_id from payload.extra_body.program_id.
  model_config.setdefault("model_kwargs", {}).setdefault("extra_body", {})["program_id"] = program_id
  model = get_model(config=model_config)
  ```

- **Program release hook**  
  Location: [`swebench.py`](src/minisweagent/run/benchmarks/swebench.py#L181) (`finally:`).  

  What: Send `POST /programs/release` to ThunderAgent with the same `program_id` after the instance finishes.  

  Why: Frees router-side bookkeeping (tokens / pause-resume state) so finished programs do not linger.

  Implementation snippet:
  ```python
  # Extract the base URL.
  base_url = (
      model.config.model_kwargs.get("api_base")
      or model.config.model_kwargs.get("base_url")
      or ""
  ).rstrip("/")
  # ThunderAgent exposes /programs/release (no /v1 prefix).
  if base_url.endswith("/v1"):
      base_url = base_url[:-3]
  if base_url:
      # Tell ThunderAgent to release this program_id to free router-side state.
      requests.post(f"{base_url}/programs/release", json={"program_id": program_id}, timeout=5)
  ```


## Repository Layout

This tree summarizes the key paths used by this guide:

- `mini-swe-agent/` contains the mini-swe-agent fork used in this guide: `scripts/` has runnable helpers (`setup/` and `reproduce/`), and `src/minisweagent/` is the Python package.


```text
mini-swe-agent/
|-- scripts/
|   |-- setup/
|   |   `-- setup.sh
|   `-- reproduce/
|       |-- reproduce_glm4.6
|       `-- reproduce_qwen3_235B
`-- src/
    `-- minisweagent/
```