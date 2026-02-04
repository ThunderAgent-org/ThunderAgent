# OpenHands (code) + ThunderAgent Use Case

## Overview
This folder contains reproducible end-to-end guides for running SWE-bench evaluations with **ThunderAgent + vLLM** across  **OpenHands (code)**.

![ThunderAgent overview](../docs/thunder/figures/thunder.jpg)

## Prerequisites
- Python 3.10 -- 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Docker daemon (SWE-bench instances run in Docker)
- GPU(s) with appropriate CUDA drivers

## Reproduction
These scripts reproduce end-to-end runs with **vLLM + ThunderAgent + OpenHands (code)**.

Environment setup: [`setup.sh`](scripts/setup/setup.sh).

- [`reproduce_glm4.6`](scripts/reproduce/reproduce_glm4.6): reproduce with GLM-4.6-FP8.
- [`reproduce_qwen3_235B`](scripts/reproduce/reproduce_qwen3_235B): reproduce with Qwen3-235B-A22B-Instruct-2507.

Expected result (throughput comparison):
![throughput_compare](../docs/openhands/figures/throughput_compare.png)

## Setup
```bash
# Create and activate env
uv venv --python 3.12
source .venv/bin/activate

# Install vLLM (GPU build) and OpenHands (code) in editable mode
uv pip install vllm --torch-backend=auto
uv pip install -e examples/inference/OpenHands
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

3) Configure `examples/inference/OpenHands/config.toml`.

4) Run SWE-Bench via OpenHands through ThunderAgent:
```bash
cd examples/inference
python -m evaluation.benchmarks.swe_bench.run_infer \
  --config-file OpenHands/config.toml \
  --llm-config vllm_local \
  --agent-cls CodeActAgent \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --max-iterations 50 \
  --eval-num-workers 128 \
  --eval-output-dir test
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

3) Configure `examples/inference/OpenHands/config.toml`.

4) Run SWE-Bench via OpenHands through ThunderAgent:
```bash
cd examples/inference
python -m evaluation.benchmarks.swe_bench.run_infer \
  --config-file OpenHands/config.toml \
  --llm-config vllm_local \
  --agent-cls CodeActAgent \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --max-iterations 50 \
  --eval-num-workers 128 \
  --eval-output-dir test
```

## What we changed in OpenHands(to reuse in your own agent workflow)
- **Program ID injection**  
  Location: [`run_infer.py`](evaluation/benchmarks/swe_bench/run_infer.py#L612) (`process_instance()`), plus [`llm.py`](openhands/llm/llm.py#L330).  

  What: For each SWE-bench instance, compute `program_id = "swe-" + sha1(f"{instance_id}:{pid}")[:16]`, store it in `OPENHANDS_PROGRAM_ID`, and inject it into every LLM request via `extra_body.program_id`.  

  Why: ThunderAgent reads `program_id` from request JSON (`payload.program_id` or `payload.extra_body.program_id`) to keep per-program routing state.  

  Implementation snippet:
  ```python
  # run_infer.py: create per-instance program_id and export it
  digest = hashlib.sha1(f"{instance_id}:{os.getpid()}".encode("utf-8")).hexdigest()
  program_id = f"swe-{digest[:16]}"
  os.environ["OPENHANDS_PROGRAM_ID"] = program_id

  # llm.py: inject into each request
  thunderagent_program_id = os.environ.get("OPENHANDS_PROGRAM_ID")
  extra_body = kwargs.get("extra_body")
  extra_body = extra_body.copy() if isinstance(extra_body, dict) else {}
  extra_body["program_id"] = thunderagent_program_id
  kwargs["extra_body"] = extra_body
  ```

- **Program release hook**  
  Location: [`run_infer.py`](evaluation/benchmarks/swe_bench/run_infer.py#L753) (`finally:`).  

  What: After the instance finishes (success or failure), send `POST /programs/release` to ThunderAgent with the same `program_id`.  

  Why: Frees router-side bookkeeping (tokens / pause-resume state) so finished programs do not linger.

  Implementation snippet:
  ```python
  # base_url is the same one used for LLM calls (usually ends with /v1)
  # POST {base_url_without_/v1}/programs/release with {"program_id": program_id}
  _release_thunderagent_program(metadata.llm_config.base_url, program_id)
  ```

## Repository Layout

This tree summarizes the key paths used by this guide:
- `OpenHands/` contains the OpenHands (code) checkout used in this guide: `config.toml` is the ThunderAgent+vLLM LLM config, `scripts/` has runnable helpers (`setup/` and `reproduce/`), and `evaluation/benchmarks/swe_bench/` is the SWE-bench harness entry.


```text
OpenHands/
|-- config.toml
|-- scripts/
|   |-- setup/
|   |   `-- setup.sh
|   `-- reproduce/
|       |-- reproduce_glm4.6
|       `-- reproduce_qwen3_235B
|-- evaluation/
|   `-- benchmarks/
|       `-- swe_bench/
`-- openhands/
    `-- llm/
        `-- llm.py
```