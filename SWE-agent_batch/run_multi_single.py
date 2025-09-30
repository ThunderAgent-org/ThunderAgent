# scripts/run_multiple_single.py
from pathlib import Path
import yaml

from sweagent.run.run_single import run_from_config, RunSingleConfig
from sweagent.agent.models import LocalVLLMOfflineModel

# 可选：提前预热一次，避免第一次 run_single 花时间加载
# 如果你的 config 已经写在 yaml 里，可以改为先解析 config，再从 agent 里取 model_path。
# 这里只演示直接构造一次。
# from sweagent.agent.models import LocalVLLMOfflineModelConfig
# from sweagent.tools.tools import ToolConfig
# preload_config = LocalVLLMOfflineModelConfig(
#     name="local_vllm_offline",
#     model_path="/data/models/GLM-4.5-FP8",
#     tensor_parallel_size=8,
# )
# LocalVLLMOfflineModel(preload_config, ToolConfig())

def load_config(path: Path) -> RunSingleConfig:
    data = yaml.safe_load(path.read_text())
    return RunSingleConfig.model_validate(data)

def main():
    config_paths = [
        Path("config/local_vllm_offline_run1.yaml"),
        Path("config/local_vllm_offline_run2.yaml"),
        # 你可以添加更多 run single 的配置文件
    ]

    configs = [load_config(path) for path in config_paths]

    for cfg in configs:
        run_from_config(cfg)

if __name__ == "__main__":
    main()