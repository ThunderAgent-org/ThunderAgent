# Examples

ThunderAgent includes end-to-end examples for both inference/rollout and RL training pipelines.

## Inference / Rollout

Complete inference examples demonstrating ThunderAgent with various agentic frameworks:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **SWE-Agent** | `examples/inference/mini-swe-agent` | Software engineering agent using [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with Docker-based code sandboxes |
| **OpenHands** | `examples/inference/OpenHands` | General software development and science discovery agent using [OpenHands](https://github.com/All-Hands-AI/OpenHands) |
| **ToolOrchestra** | `examples/inference/ToolOrchestra` | Multi-tool orchestration agent for complex workflows |

Each example includes:
- Setup scripts and configuration
- Instructions for single-node and multi-node deployments
- Integration with ThunderAgent's program-aware scheduling

### Quick Example: mini-swe-agent

```bash
# 1. Start vLLM backend
vllm serve Qwen/Qwen3-32B --port 8000

# 2. Start ThunderAgent
thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000 --router tr --metrics --profile

# 3. Run mini-swe-agent pointing to ThunderAgent
cd examples/inference/mini-swe-agent
# Follow README.md for setup and execution
```

## RL Training

RL training pipelines that use ThunderAgent as the inference backend:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **Search-R1 Agent** | `examples/rl_training/SkyRL` | RL training for search-augmented reasoning using [SkyRL](https://github.com/NovaSky-AI/SkyRL) |
| **SWE Agent** | `examples/rl_training/SkyRL` | RL training for software engineering agent |

The RL training examples demonstrate:
- Integration with [slime](https://github.com/THUDM/slime) and [SkyRL](https://github.com/NovaSky-AI/SkyRL) training frameworks
- Using ThunderAgent with the `skyrl` backend type for multi-engine scheduling
- Docker-based training environments

### Documentation

Additional setup documentation and guides are available in `examples/inference/docs/` organized by agent type:
- `examples/inference/docs/thunder/` -- ThunderAgent-specific setup
- `examples/inference/docs/mini-swe-agent/` -- SWE-Agent setup
- `examples/inference/docs/openhands/` -- OpenHands setup
- `examples/inference/docs/toolorchestra/` -- ToolOrchestra setup
