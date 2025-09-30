# test_function_calling.py
import os
import json
from vllm import LLM, SamplingParams
from pprint import pprint
import sys, time
from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
# 建议单机先关 IB，稳定后再恢复（可按需删除）
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

llm = None
llm = LLM(
    model="/data/models/GLM-4.5-FP8",
    tokenizer="/data/models/GLM-4.5-FP8",
    tensor_parallel_size=8,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)


tools = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute bash commands",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"}
            },
            "required": ["command"]
        }
    }
}]

messages = [{"role": "user", "content": "List the files in the current directory"}]
request = ChatCompletionRequest(
    model="/data/models/GLM-4.5-FP8",
    messages=messages,
    tools=tools
)
out = llm.chat(messages, tools=tools, sampling_params=SamplingParams(temperature=0, max_tokens=200))
print(out)
parser = Glm4MoeModelToolParser(tokenizer=llm.get_tokenizer())

parsed = parser.extract_tool_calls(out[0].outputs[0].text, request=request)

print([tool.model_dump() for tool in parsed.tool_calls])


if llm is not None:
    close = getattr(llm, "shutdown", None) or getattr(llm, "_shutdown", None)
    if callable(close):
        close()
    del llm
