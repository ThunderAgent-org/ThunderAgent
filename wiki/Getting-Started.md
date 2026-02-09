# Getting Started

## Prerequisites

- Python >= 3.10
- An inference backend: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), or [SkyRL](https://github.com/NovaSky-AI/SkyRL)

## Installation

Install ThunderAgent from source:

```bash
git clone git@github.com:HaoKang-Timmy/ThunderAgent.git
cd ThunderAgent
pip install -e .
```

Dependencies (installed automatically):
- `fastapi`
- `httpx`
- `uvicorn`

## Quick Start

### 1. Start an inference backend

Choose a backend and serve a model. For example, with vLLM:

```bash
uv pip install vllm --torch-backend=auto
vllm serve Qwen/Qwen3-32B --port 8000
```

Or with SGLang:

```bash
pip install sglang
python -m sglang.launch_server --model Qwen/Qwen3-32B --port 8000
```

### 2. Launch ThunderAgent

```bash
thunderagent \
  --backend-type vllm \
  --backends http://localhost:8000 \
  --port 9000 \
  --metrics \
  --profile
```

ThunderAgent now listens on port 9000 and proxies requests to the vLLM backend on port 8000.

### 3. Send requests through ThunderAgent

Point your agent client to ThunderAgent (port 9000) instead of the backend directly. Add `program_id` to identify each agentic workflow:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:9000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"program_id": "my-agent-session-1"},
)
print(response.choices[0].message.content)
```

## Integration with Existing Agentic Workflows

The only change required is adding `program_id` to the `extra_body` of your OpenAI API calls:

```python
# Before (direct to backend)
openai.client.chat.completions.create(
    model=self.config.model_name,
    messages=messages,
)

# After (through ThunderAgent)
extra_body = {"program_id": "unique_id"}
openai.client.chat.completions.create(
    model=self.config.model_name,
    messages=messages,
    extra_body=extra_body,
)
```

If your agentic workflow uses Docker containers, you can also pass Docker container IDs:

```python
extra_body = {
    "program_id": "unique_id",
    "docker_ids": ["container_id_1", "container_id_2"],
}
```

## Multi-Backend Setup

ThunderAgent supports multiple backends for load balancing and capacity scheduling:

```bash
thunderagent \
  --backend-type vllm \
  --backends http://gpu1:8000,http://gpu2:8000,http://gpu3:8000 \
  --port 9000 \
  --router tr \
  --metrics
```

The `tr` router mode enables program-aware capacity scheduling across all backends.

## Running as a Python Module

You can also run ThunderAgent directly as a Python module:

```bash
python -m ThunderAgent \
  --backend-type vllm \
  --backends http://localhost:8000 \
  --port 9000
```
