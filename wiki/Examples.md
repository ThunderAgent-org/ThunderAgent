# Examples

ThunderAgent includes end-to-end examples for both inference/rollout and RL training pipelines.

## Serving

Complete inference examples demonstrating ThunderAgent with various agentic frameworks:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **SWE-Agent** | `examples/inference/mini-swe-agent` | Software engineering agent using [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with Docker-based code sandboxes |
| **OpenHands** | `examples/inference/OpenHands` | General software development and science discovery agent using [OpenHands](https://github.com/All-Hands-AI/OpenHands) |
| **ToolOrchestra** | `examples/inference/ToolOrchestra` | Routing agent that orchestrates other sub-agents using [ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) |

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

RL training pipelines that integrate ThunderAgent into the rollout phase for program-aware scheduling:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **Search Agent** | `examples/rl_training/slime` | RL training for search-augmented reasoning using [slime](https://github.com/THUDM/slime) |
| **SWE Agent** | `examples/rl_training/SkyRL` | RL training for software engineering agent using [SkyRL](https://github.com/NovaSky-AI/SkyRL) |

The RL training examples demonstrate:
- Integrating ThunderAgent's router into the rollout phase of [SkyRL](https://github.com/NovaSky-AI/SkyRL) (which uses vLLM's async LLM engine internally) and [slime](https://github.com/THUDM/slime) (which uses sglang inference engine) training frameworks
- Using the `--backend-type skyrl` option to adapt to SkyRL's aggregated JSON metrics format
- Docker-based training environments
