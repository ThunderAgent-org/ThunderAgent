# mini-swe-agent + ThunderAgent Use Case

## Prerequisites
- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- GPU(s) with appropriate CUDA drivers

## Intro
Run the official [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on SWE-Bench while routing model traffic through ThunderAgent to manage vLLM capacity and per-task program tracking.

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

## How to run the experiment

### One node example
1) Start vLLM to serve the model:
```bash
vllm serve <MODEL_NAME> --tensor-parallel-size <NUM_GPUS> --enable-auto-tool-choice --tool-call-parser <TOOL_PARSER> --port <VLLM_PORT>
```
2) Start ThunderAgent (pointing at your vLLM backend):
```bash
python -m ThunderAgent --backends http://localhost:<VLLM_PORT> --port <TA_PORT>
```
3) Configure mini-swe-agent to call ThunderAgent:

- Edit `examples/inference/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml`
- Set `model.model_kwargs.api_base` to `http://localhost:<TA_PORT>/v1`

4) Run SWE-Bench via mini-swe-agent through ThunderAgent:
```bash
mini-extra swebench \
  --subset lite \
  --split test \
  --workers 128 \
  --output ./swebench_output 
```

### multi nodes example
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
3) Configure mini-swe-agent to call ThunderAgent:

- Edit `examples/inference/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml`
- Set `model.model_kwargs.api_base` to `http://<TA_HOST>:<TA_PORT>/v1`

4) Run SWE-Bench via mini-swe-agent through ThunderAgent:
```bash
mini-extra swebench \
  --subset lite \
  --split test \
  --workers 128 \
  --output ./swebench_output 
```


## What we changed (to reuse in your own setup)
- **Program ID injection**  
  Location: `examples/inference/mini-swe-agent/src/minisweagent/run/benchmarks/swebench.py` in `process_instance()`.  
  What: Generate a unique `program_id` per instance and inject it into `model_kwargs.extra_body.program_id` before creating the model.  
  Why: `LitellmModel` forwards `model_kwargs` to `litellm.completion(...)`, and ThunderAgent extracts `program_id` from either `payload["program_id"]` or `payload["extra_body"]["program_id"]` (see `ThunderAgent/app.py:get_program_id()`), so each instance gets an isolated ProgramState.
  ```python
  program_id = f"swe-{next(_PROGRAM_COUNTER):06d}"
  model_config.setdefault("model_kwargs", {}).setdefault("extra_body", {})["program_id"] = program_id
  ```

- **Program release hook (free router-side state when an instance finishes)**  
  Location: `examples/inference/mini-swe-agent/src/minisweagent/run/benchmarks/swebench.py`, `process_instance()` -> `finally:`.  
  What: After saving results, call `POST /programs/release` on ThunderAgent with the same `program_id`.  
  Why: ThunderAgent tracks per-program state (tokens, pause/resume bookkeeping). Releasing prevents finished programs from lingering and affecting scheduling/metrics.
  ```python
  base_url = (
      model.config.model_kwargs.get("api_base")
      or model.config.model_kwargs.get("base_url")
      or ""
  ).rstrip("/")
  if base_url.endswith("/v1"):
      base_url = base_url[:-3]
  requests.post(f"{base_url}/programs/release", json={"program_id": program_id}, timeout=5)
  ```
