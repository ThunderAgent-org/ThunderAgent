# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import json
import requests
import asyncio
import subprocess
import traceback
import uuid
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import lru_cache
import sys

# HuggingFace Hub fast-download mode (`HF_HUB_ENABLE_HF_TRANSFER=1`) requires `hf_transfer`.
# If enabled but missing, transformers/huggingface_hub will raise at download time.
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") in {"1", "true", "True"}:
    try:
        import hf_transfer  # type: ignore[unused-ignore]  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(
            "[eval_hle_local] WARNING: HF_HUB_ENABLE_HF_TRANSFER=1 but 'hf_transfer' is not installed; "
            "disabling fast downloads. Install with `pip install hf_transfer` to re-enable.",
            flush=True,
        )
REPO_PATH = os.getenv("REPO_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from LLM_CALL import get_llm_response
import multiprocessing as mp
import argparse

# HLE profiling / structured logging (tau2-bench style)
from hle_logging import (
    configure_hle_logging,
    log_profile_event,
    log_user_judge_event,
    get_hle_logger,
    set_task_context,
    set_step_context,
    clear_task_context,
    task_context,
)

MODEL_NAME = None
my_output_dir = None
MAX_ROUNDS = None
MODEL_TYPE = None
MODEL_MAPPING = None
TOOL_PRICING = None
vllm_model_configs = None
with open('tools.json') as f:
    raw_tools = json.load(f)

HLE_TOKENIZER_NAME = os.environ.get("HLE_TOKENIZER_NAME") or "Qwen/Qwen3-8B"


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(HLE_TOKENIZER_NAME)

# Together routing for open-source expert models (see /workspace/HLE-expert-lm.md)
# IMPORTANT: Together dedicated endpoints must be called using the endpoint **Name** (e.g. "HK123/..."),
# not the endpoint ID ("endpoint-...").
TOGETHER_SERVERLESS_MAP = {
    # Serverless model: pass Together's model name directly.
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}
TOGETHER_DEDICATED_NAME_MAP = {
    # Dedicated endpoints: use the "Name" field from Together UI/CLI (owner-prefixed).
    "Qwen/Qwen2.5-Coder-32B-Instruct": "HK123/Qwen/Qwen2.5-Coder-32B-Instruct-afa30e99",
    "Qwen/Qwen2.5-Math-7B-Instruct": "HK123/Qwen/Qwen2.5-Math-7B-Instruct-edb7ac78",
    "Qwen/Qwen2.5-Math-72B-Instruct": "HK123/Qwen2.5-Math-72B-Instruct-1d5fc83c",
}


def _together_model_for(model_name: str) -> str:
    if model_name in TOGETHER_DEDICATED_NAME_MAP:
        return TOGETHER_DEDICATED_NAME_MAP[model_name]
    if model_name in TOGETHER_SERVERLESS_MAP:
        return TOGETHER_SERVERLESS_MAP[model_name]
    return model_name

MODEL_MAPPING = {
    # All reasoners → gpt-5 (OpenAI)
    "reasoner-1": "gpt-5",
    "reasoner-2": "gpt-5",
    "reasoner-3": "gpt-5",

    # All search → gpt-5-mini (OpenAI)
    "search-1": "gpt-5-mini",
    "search-2": "gpt-5-mini",
    "search-3": "gpt-5-mini",

    # All answer (including math) → Together-hosted gpt-oss-120b
    "answer-1": "openai/gpt-oss-120b",
    "answer-2": "openai/gpt-oss-120b",
    "answer-3": "openai/gpt-oss-120b",
    "answer-4": "openai/gpt-oss-120b",
    "answer-math-1": "openai/gpt-oss-120b",
    "answer-math-2": "openai/gpt-oss-120b",
}
TOOL_PRICING = {
    "gpt-5": {
        "input_tokens_per_million": 1.25/10000000,
        "output_tokens_per_million": 10/1000000
    },
    "gpt-5-mini": {
        "input_tokens_per_million": 0.25/10000000,
        "output_tokens_per_million": 2/1000000
    },
    "openai/gpt-oss-120b": {
        "input_tokens_per_million": 0.3/1000000,
        "output_tokens_per_million": 0.3/1000000
    },
    "Qwen/Qwen3-32B": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen3-8B": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "code_interpreter_per_second": 0.0000083,
    "tavily": {
        "search": 0.01,
        "extract": 0.002
    },
}
ALL_TOOLS = {
    "enhance_reasoning": {
        'model': ["reasoner-1", "reasoner-2", "reasoner-3"]
    },
    "answer": {
        'model': ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]
    },
    "search": {
        "model": ["search-1", "search-2", "search-3"]
    },
}

def cut_seq(seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    tokenizer = get_tokenizer()
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def call_tool(arguments):
    tool_name = arguments.get("tool")
    eid = arguments.get("eid")

    with task_context(
        task_id=str(arguments.get("id", "unknown")),
        domain="hle",
        eid=eid if isinstance(eid, int) else None,
    ):
        set_step_context(arguments.get("step"))
        t0 = time.perf_counter()
        try:
            # Keep original structure from eval_hle.py, but with local expert routing + profiling.
            _ = time.time()  # start_time (legacy; kept for compatibility)

            if arguments["tool"] == "enhance_reasoning":
                supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS["enhance_reasoning"]["model"]]
                assert (
                    arguments["model"] in supported_models
                ), f"Model {arguments['model']} is not supported in enhance_reasoning. Support models: {supported_models}"

                prompt = arguments["context_str"].strip() + "\n\n"
                prompt += (
                    f"Question: {arguments['problem']}\n"
                    "Instead of directly answering the question, please write additional python code that will give intermidiate results after execution. "
                    "Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
                )
                model_name = arguments["model"]
                arguments["messages"] = prompt

                llm_t0 = time.perf_counter()
                if "gpt-5" in model_name.lower():
                    response = get_llm_response(
                        model=model_name,
                        messages=prompt,
                        return_raw_response=True,
                        max_length=40000,
                        temperature=1,
                    )
                else:
                    response = get_llm_response(
                        model=model_name,
                        messages=prompt,
                        return_raw_response=True,
                        model_type="vllm",
                        max_length=8000,
                        temperature=0.2,
                        model_config=arguments["vllm_model_configs"][model_name],
                        model_config_path=arguments["vllm_model_configs"]["vllm_model_config_path"],
                        model_config_idx=arguments["eid"],
                    )
                arguments["_llm_ms"] = (time.perf_counter() - llm_t0) * 1000.0

                if isinstance(response, str):
                    arguments["generated_code"] = ""
                    arguments["exec_result"] = ""
                    return arguments

                try:
                    generated_code = response.choices[0].message.content.split("```python")[-1].split("```")[0]
                except Exception:
                    generated_code = ""

                if generated_code == "":
                    arguments["generated_code"] = ""
                    arguments["exec_result"] = ""
                    return arguments

                code_path = str(os.path.join(arguments["cur_output_dir"], f'exec_code_{arguments["id"]}.py'))
                with open(code_path, "w") as f:
                    f.write(generated_code)

                exec_result = ""
                exec_start = time.time()
                try:
                    run_ret = subprocess.run(["python", code_path], timeout=60, capture_output=True, text=True)
                    exec_result = run_ret.stdout
                    with open(os.path.join(arguments["cur_output_dir"], f'exec_out_{arguments["id"]}.txt'), "w") as f:
                        f.write(exec_result)
                except Exception:
                    pass
                _ = time.time() - exec_start  # exec_time (legacy)

                arguments["generated_code"] = generated_code
                arguments["exec_result"] = exec_result
                return arguments

            if arguments["tool"] == "answer":
                prompt = arguments["context_str"].strip() + "\n\nProblem:\n" + arguments["problem"]
                response_str = ""
                pred = ""

                # Measure ONLY the expert model portion (exclude evaluator call).
                expert_t0 = time.perf_counter()

                if "qwen3" in arguments["model"].lower():
                    model_name = arguments["model"]
                    prompt2 = (
                        prompt
                        + "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
                    )
                    arguments["messages"] = prompt2
                    response = get_llm_response(
                        model=model_name,
                        messages=prompt2,
                        return_raw_response=True,
                        model_type="vllm",
                        max_length=8000,
                        temperature=0.2,
                        model_config=arguments["vllm_model_configs"][model_name],
                        model_config_path=arguments["vllm_model_configs"]["vllm_model_config_path"],
                        model_config_idx=arguments["eid"],
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    if isinstance(response, str):
                        arguments["response"] = ""
                        arguments["pred"] = ""
                        arguments["correctness"] = False
                        return arguments
                    response_str = response.choices[0].message.content
                    if isinstance(response_str, str):
                        pred = response_str.split("<answer>")[-1].split("</answer>")[0].strip()
                    else:
                        pred = ""

                elif "qwen2.5-math" in arguments["model"].lower():
                    model_name = arguments["model"]
                    messages = [
                        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                        {"role": "user", "content": prompt},
                    ]
                    arguments["messages"] = messages
                    response = get_llm_response(
                        model=_together_model_for(model_name),
                        messages=messages,
                        return_raw_response=True,
                        model_type="nv/dev",
                        max_length=2000,
                        temperature=0.2,
                    )
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    if isinstance(response, str):
                        arguments["response"] = ""
                        arguments["pred"] = ""
                        arguments["correctness"] = False
                        return arguments
                    response_str = response.choices[0].message.content
                    if not isinstance(response_str, str) or "\\boxed{" not in response_str:
                        pred = ""
                    else:
                        pred_components = response_str.split("\\boxed{")[-1].split("}")[:-1]
                        pred = "}".join(pred_components).strip()

                elif "gpt-5" in arguments["model"].lower():
                    model_name = arguments["model"]
                    prompt2 = prompt + (
                        "\n\nTake a deep breath and think hard with high reasoning, wrap the thoughts within <think> and </think>, "
                        "and wrap only the exact answer without any explanation within <answer> and </answer>."
                        "Output using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
                    )
                    arguments["messages"] = prompt2
                    response = get_llm_response(model=model_name, messages=prompt2, return_raw_response=True, max_length=40000)
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    if isinstance(response, str):
                        arguments["response"] = ""
                        arguments["pred"] = ""
                        arguments["correctness"] = False
                        return arguments
                    response_str = response.choices[0].message.content
                    if isinstance(response_str, str):
                        pred = response_str.split("<answer>")[-1].split("</answer>")[0].strip()
                    else:
                        pred = ""

                elif "gpt-oss-120b" in arguments["model"].lower():
                    model_name = arguments["model"]
                    prompt2 = (
                        prompt
                        + "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
                    )
                    arguments["messages"] = prompt2
                    response = get_llm_response(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt2}],
                        return_raw_response=True,
                        model_type="together",
                        max_length=8000,
                        temperature=0.2,
                    )
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    if isinstance(response, str):
                        arguments["response"] = ""
                        arguments["pred"] = ""
                        arguments["correctness"] = False
                        return arguments
                    response_str = response.choices[0].message.content
                    if isinstance(response_str, str):
                        pred = response_str.split("<answer>")[-1].split("</answer>")[0].strip()
                    else:
                        pred = ""

                elif "llama-3.3" in arguments["model"].lower():
                    model_name = arguments["model"]
                    prompt2 = (
                        prompt
                        + "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
                    )
                    arguments["messages"] = prompt2
                    response = get_llm_response(
                        model=_together_model_for(model_name),
                        messages=[{"role": "user", "content": prompt2}],
                        return_raw_response=True,
                        model_type="nv/dev",
                        max_length=40000,
                        temperature=0.2,
                    )
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    if isinstance(response, str):
                        arguments["response"] = ""
                        arguments["pred"] = ""
                        arguments["correctness"] = False
                        return arguments
                    response_str = response.choices[0].message.content
                    if isinstance(response_str, str):
                        pred = response_str.split("<answer>")[-1].split("</answer>")[0].strip()
                    else:
                        pred = ""

                else:
                    arguments["_expert_ms"] = (time.perf_counter() - expert_t0) * 1000.0
                    arguments["response"] = ""
                    arguments["pred"] = ""
                    arguments["correctness"] = False
                    return arguments

                if pred.strip() == "" or len(pred.split(" ")) > 500:
                    correctness = False
                elif pred.strip().lower() == arguments["answer"].strip().lower():
                    correctness = True
                else:
                    # Throughput experiments: judge disabled by default (exact match only).
                    enable_judge = os.getenv("HLE_ENABLE_JUDGE", "0").strip().lower() in {"1", "true", "yes"}
                    judge_model = (os.getenv("HLE_JUDGE_MODEL", "gpt-5-mini") or "gpt-5-mini").strip()
                    if enable_judge:
                        eval_prompt = (
                            f"Question: {arguments['problem']}\n\n"
                            f"Student answer: {pred}\n\n"
                            f"Reference answer: {arguments['answer']}\n\n"
                            "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer."
                        )
                        judge_t0 = time.perf_counter()
                        eval_response = get_llm_response(model=judge_model, messages=eval_prompt, temperature=1)
                        judge_ms = (time.perf_counter() - judge_t0) * 1000.0
                        arguments["_judge_ms"] = judge_ms
                        log_user_judge_event("evaluator", model=judge_model, duration_ms=judge_ms)
                        eval_result = eval_response.split("<correct>")[-1].split("</correct>")[0]
                        correctness = eval_result.lower() == "true"
                    else:
                        correctness = False

                arguments["response"] = response_str
                arguments["pred"] = pred
                arguments["correctness"] = correctness
                return arguments

            if arguments["tool"] == "search":
                contents = []
                prompt = arguments["context_str"].strip() + "\n\n"
                prompt += (
                    f"Question: {arguments['problem']}\n"
                    "Instead of directly answering the question, please write a query to search for a piece of relevant and missing information. "
                    "The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."
                )
                cur_query_writer = arguments["model"]
                query_to_call = None

                query_llm_t0 = time.perf_counter()
                response = get_llm_response(
                    model=cur_query_writer,
                    messages=[{"role": "user", "content": prompt}],
                    return_raw_response=True,
                    max_length=4000,
                    temperature=(1 if "gpt-5" in cur_query_writer.lower() else 0.2),
                )
                arguments["_query_llm_ms"] = (time.perf_counter() - query_llm_t0) * 1000.0
                if isinstance(response, str):
                    query_to_call = arguments["problem"]
                else:
                    try:
                        query_to_call = response.choices[0].message.content.split("<query>")[-1].split("</query>")[0]
                    except Exception:
                        query_to_call = arguments["problem"]

                if query_to_call is not None and len(query_to_call) >= 5:
                    payload = {"queries": [query_to_call[:390]], "topk": 50, "return_scores": True, "eid": arguments["id"]}
                    results = None
                    all_vllm_model_configs = arguments["vllm_model_configs"]
                    retrieval_t0 = time.perf_counter()
                    retrieval_retries = 0
                    max_attempts = int(os.getenv("HLE_RETRIEVAL_MAX_ATTEMPTS", "8"))
                    connect_timeout_s = float(os.getenv("HLE_RETRIEVAL_CONNECT_TIMEOUT_S", "3"))
                    read_timeout_s = float(os.getenv("HLE_RETRIEVAL_READ_TIMEOUT_S", "60"))
                    base_backoff_s = float(os.getenv("HLE_RETRIEVAL_BACKOFF_S", "2"))
                    for attempt in range(max_attempts):
                        try:
                            endpoints = all_vllm_model_configs.get("retrieval") or []
                            if not endpoints:
                                raise RuntimeError("no_retrieval_endpoints_configured")
                            cur_model_config = random.choice(endpoints)
                            resp = requests.post(
                                f'http://{cur_model_config["ip_addr"]}:{cur_model_config["port"]}/retrieve',
                                json=payload,
                                timeout=(connect_timeout_s, read_timeout_s),
                            )
                            resp.raise_for_status()
                            results = resp.json()
                            break
                        except Exception:
                            retrieval_retries += 1
                            time.sleep(base_backoff_s * (attempt + 1))
                            continue
                    arguments["_retrieval_ms"] = (time.perf_counter() - retrieval_t0) * 1000.0
                    arguments["_retrieval_retries"] = retrieval_retries

                    if results:
                        # Distinguish local-only retrieval vs Tavily fallback (if enabled on retrieval server).
                        used_tavily = False
                        local_hits = 0
                        tavily_hits = 0
                        try:
                            if isinstance(results, list) and results and isinstance(results[0], list):
                                for rr in results[0]:
                                    if not isinstance(rr, dict):
                                        continue
                                    src = rr.get("source")
                                    if src == "tavily" or (
                                        isinstance(rr.get("score"), (int, float)) and float(rr.get("score")) < 0
                                    ):
                                        used_tavily = True
                                        tavily_hits += 1
                                    elif src == "local":
                                        local_hits += 1
                        except Exception:
                            pass
                        arguments["_search_backend"] = "tavily_fallback" if used_tavily else "local_only"
                        arguments["_search_local_hits"] = local_hits
                        arguments["_search_tavily_hits"] = tavily_hits

                        for r in results[0]:
                            if "content" in r["document"]:
                                contents.append(r["document"]["content"])
                            elif "contents" in r["document"]:
                                contents.append(r["document"]["contents"])

                arguments["query"] = query_to_call
                arguments["search_results_data"] = contents
                if "tokenizer" in arguments:
                    arguments.pop("tokenizer")
                return arguments

            # Unknown tool: mark as error but do not raise.
            arguments["tool_error"] = f"unknown_tool:{tool_name}"
            arguments["tool_error_type"] = "unknown_tool"
            return arguments

        except Exception as e:
            # Never raise from tool threads; bubble errors back to run_single so it can mark task complete.
            try:
                arguments["tool_error"] = repr(e)
                arguments["tool_error_type"] = type(e).__name__
                arguments["tool_error_traceback"] = traceback.format_exc()
            except Exception:
                pass
            return arguments
        finally:
            dur_ms_total = (time.perf_counter() - t0) * 1000.0
            if tool_name == "answer" and arguments.get("_expert_ms") is not None:
                log_profile_event(
                    "tool_call",
                    tool=tool_name,
                    model=arguments.get("model"),
                    duration_ms=float(arguments.get("_expert_ms") or 0.0),
                    duration_total_ms=dur_ms_total,
                    judge_ms=float(arguments.get("_judge_ms") or 0.0) if arguments.get("_judge_ms") is not None else None,
                )
            elif tool_name == "search":
                log_profile_event(
                    "tool_call",
                    tool=tool_name,
                    model=arguments.get("model"),
                    duration_ms=dur_ms_total,
                    search_backend=arguments.get("_search_backend"),
                    search_local_hits=arguments.get("_search_local_hits"),
                    search_tavily_hits=arguments.get("_search_tavily_hits"),
                )
            else:
                log_profile_event("tool_call", tool=tool_name, model=arguments.get("model"), duration_ms=dur_ms_total)

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Any, Callable

# task_list is an iterable of (func, arg) pairs
async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    concurrency: int = 2,
    progress: bool = False,
    return_exceptions: bool = False,
):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    # create the executor sized to your concurrency gate
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # wrap each task so it obeys the semaphore
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                try:
                    if asyncio.iscoroutinefunction(func):
                        res = await func(arg)
                    else:
                        res = await loop.run_in_executor(executor, func, arg)
                    return idx, res, None
                except Exception as e:
                    return idx, None, e

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a))
                 for i, (f, a) in enumerate(task_list)]

        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks))
        else:
            pbar = None

        try:
            # update progress as tasks complete
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                else:
                    if return_exceptions:
                        results[idx] = err
                    else:
                        # cancel remaining, then re-raise the first error
                        for t in tasks:
                            t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise err
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        return results

def run_single(e):
    set_task_context(task_id=str(e.get("id", "unknown")), domain="hle", eid=e.get("eid") if isinstance(e.get("eid"), int) else None)
    logger = get_hle_logger()
    out_path = os.path.join(my_output_dir, f"{e['id']}.json")
    if os.path.isfile(out_path):
        logger.info(f"[HLE_TASK_COMPLETE] id={e['id']} status=skipped")
        clear_task_context()
        return {"id": e["id"], "skipped": True}
    doc_list = []
    code_list = []
    attempt_list = []
    exp_start_time = time.time()
    problem = e['question']
    user_problem = problem
    answer = e['answer']
    all_tool_calls = []
    final_correct = False
    final_answer_model = None
    final_pred = ''
    task_status = "done"
    task_error_type = None
    all_tool_responses = {}
    all_message_responses = {}
    used_tools = []
    program_id = f"hle:{e['id']}:{uuid.uuid4().hex[:8]}"
    router_url = (os.getenv("ROUTER_URL") or "").strip()

    def _release_program() -> None:
        if not router_url:
            return
        try:
            requests.post(f"{router_url.rstrip('/')}/programs/release", json={"program_id": program_id}, timeout=2.0)
        except Exception:
            return

    for step in range(MAX_ROUNDS):
        set_step_context(step)
        cur_output_dir = os.path.join(my_output_dir,f"step_{step}")
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            try:
                os.makedirs(os.path.join(cur_output_dir,'tool_return'))
            except:
                pass
        tools = []
        for t in raw_tools:
            tools.append(t)
        doc_str = ''
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:1200]} ...\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
        str_cut = cut_seq(seq=attempt_str,l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str,l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        problem_length = len(get_tokenizer()(problem)['input_ids'])
        if code_attempt_str_len<27000-problem_length:
            if code_attempt_str:
                context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=27000-problem_length)
            else:
                context_str = cut_seq(seq=doc_str,l=27000-problem_length)
            context_str = context_str['string_after_cut']
            if len(doc_str)>0:
                doc_flag = True
                context_str = 'Documents:\n'+context_str
        else:
            context_str = code_attempt_str

        removed_tool = None
        if len(used_tools)>1 and used_tools[-1]==used_tools[-2]:
            updated_tools = []
            removed_tool = used_tools[-1]
            for t in tools:
                if t['function']['name']!=used_tools[-1]:
                    updated_tools.append(t)
        else:
            updated_tools = tools
        cur_tool_set = [t['function']['name'] for t in updated_tools]
        chat = [
                    {"role": "system", "content": "You are good at using tools."},
                    {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool.'"}
                ]
        orch_t0 = time.perf_counter()
        response = get_llm_response(
            model=MODEL_NAME,
            messages=chat,
            return_raw_response=True,
            model_type='vllm',
            model_config=vllm_model_configs[MODEL_NAME],
            temperature=1,
            max_length=12000,
            tools=tools,
            model_config_path=vllm_model_configs['vllm_model_config_path'],
            model_config_idx=e['eid'],
            tau2_stream_profile=True,
            extra_body={
                # Baseline/continuum may ignore these; TR-new uses program_id for scheduling.
                "program_id": program_id,
                "job_id": program_id,
                "is_last_step": bool(step == MAX_ROUNDS - 1),
            },
        )
        orch_ms = (time.perf_counter() - orch_t0) * 1000.0
        if not isinstance(response, str):
            log_profile_event(
                "llm_call",
                model=MODEL_NAME,
                backend="vllm",
                duration_ms=orch_ms,
                toolorchestra_vllm_infer_ms=getattr(response, "tau2_vllm_infer_ms", None),
                toolorchestra_vllm_prefill_ms=getattr(response, "tau2_vllm_prefill_ms", None),
                toolorchestra_vllm_decode_ms=getattr(response, "tau2_vllm_decode_ms", None),
                toolorchestra_vllm_prefill_len=getattr(response, "tau2_vllm_prompt_tokens", None),
                toolorchestra_vllm_decode_len=getattr(response, "tau2_vllm_completion_tokens", None),
            )
        else:
            log_profile_event(
                "llm_call",
                model=MODEL_NAME,
                backend="vllm",
                duration_ms=orch_ms,
                error="string_response",
            )
        cache_idx = 0
        while os.path.isfile(f"input_output/{cache_idx}.json"):
            cache_idx += 1
        if isinstance(response,str):
            continue
        tool_calls = response.choices[0].message.tool_calls or []
        cache_tool_calls = []
        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            tool_arguments = None
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except Exception:
                tool_arguments = None
            cache_tool_calls.append({
                'tool_name': tool_name,
                'tool_arguments': tool_arguments
            })
        message_dict = {
            'content': response.choices[0].message.content,
            'tool_calls': cache_tool_calls
        }
        if len(tool_calls)==0:
            all_tool_calls.append(f'342 invalid tool calls {tool_calls}')
            continue
        tool_call_list = []
        cur_tool_calls = []
        processed_tools = set()
        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            tool_arguments = None
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except Exception:
                tool_arguments = None
            if not tool_name in ALL_TOOLS:
                cur_tool_calls.append(f'350 invalid tool calls {tool_calls}')
                continue
            func_signature = ALL_TOOLS[tool_name]
            if not isinstance(tool_arguments, dict):
                cur_tool_calls.append(f'351 invalid tool call args (not json object) {tool_calls}')
                continue
            valid_tool_call = True
            for parameter_name,parameter_values in func_signature.items():
                if (not parameter_name in tool_arguments):
                    valid_tool_call = False
                    continue
                if (parameter_values!='any') and (not tool_arguments[parameter_name] in parameter_values):
                    valid_tool_call = False
            if not valid_tool_call:
                cur_tool_calls.append(f'360 invalid tool calls {tool_calls}')
                continue

            if tool_name in processed_tools:
                continue
            processed_tools.add(tool_name)
            tool_call = {
                'name': tool_name,
                'arguments': tool_arguments
            }
            cur_tool_calls.append([tool_call])
            expert_model_to_call = MODEL_MAPPING[tool_arguments['model']]
            
            call_tool_argument = None
            used_tools.append(tool_name)
            if tool_name=='enhance_reasoning':
                if 'qwen2.5-coder' in expert_model_to_call.lower():
                    max_code_length = 16000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'qwen2.5-coder' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(get_tokenizer()(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'id': e['id'],
                    'eid': e['eid'],
                    'step': step,
                }
            elif tool_call['name']=='answer':
                if 'qwen2.5-math' in expert_model_to_call.lower():
                    max_code_length = 1000
                    max_context_length = 2000
                elif 'llama-3.3' in expert_model_to_call.lower():
                    max_code_length = 10000
                    max_context_length = 80000
                elif 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'gpt-5' in expert_model_to_call.lower() or 'llama' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(get_tokenizer()(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid'],
                    'step': step,
                }
            elif tool_call['name'] in ['search']:
                if 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-oss' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 120000
                else:
                    # Conservative defaults for unknown search models.
                    max_code_length = 12000
                    max_context_length = 24000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    if 'gpt-5' in expert_model_to_call.lower():
                        doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                    else:
                        doc_str += f"Doc {doc_idx+1}: {doc[:1000]}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(get_tokenizer()(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid'],
                    'step': step,
                }
            tool_call_list.append([call_tool,call_tool_argument])
            break
        all_tool_calls.append(cur_tool_calls)

        cache_argument = []
        for t in tool_call_list:
            cache_argument.append(t[1])
        if len(tool_call_list)==0:
            continue
        cur_responses = asyncio.run(run_all(tool_call_list))
        all_tool_responses[f"turn_{step}_response"] = cur_responses
        all_message_responses[f"turn_{step}_message"] = message_dict
        finish_flag = False
        for cur_response in cur_responses:
            if isinstance(cur_response, dict) and cur_response.get("tool_error"):
                task_status = "error"
                task_error_type = cur_response.get("tool_error_type") or "tool_error"
                finish_flag = True
                break
            if cur_response['tool']=='enhance_reasoning':
                if len(cur_response['exec_result'].strip())>0:
                    code_list.append({'code': cur_response['generated_code'], 'output': cur_response['exec_result']})
            elif cur_response['tool']=='answer':
                final_correct = cur_response['correctness']
                final_answer_model = cur_response['model']
                final_pred = cur_response['pred'].strip()
                finish_flag = True
                break
            elif cur_response['tool']=='search':
                for one_doc in cur_response['search_results_data'][::-1]:
                    if not one_doc in doc_list:
                        doc_list.append(one_doc)
        log_profile_event(
            "step_complete",
            ts_unix=time.time(),
            step=step,
            job_id=program_id,
            tool=used_tools[-1] if used_tools else None,
        )
        if finish_flag:
            break
    return_dict = {
        'id': e['id'],
        'problem': problem,
        'all_tool_calls': all_tool_calls,
        'all_tool_responses': all_tool_responses,
        'answer': answer,
        'all_message_responses': all_message_responses,
        'correct': final_correct,
        'status': task_status,
    }
    if task_error_type:
        return_dict["error_type"] = task_error_type
    with open(os.path.join(my_output_dir,f"{e['id']}.json"),'w') as f:
        json.dump(return_dict,f,indent=2)
    if task_error_type:
        logger.info(f"[HLE_TASK_COMPLETE] id={e['id']} status={task_status} correct={final_correct} error_type={task_error_type}")
    else:
        logger.info(f"[HLE_TASK_COMPLETE] id={e['id']} status={task_status} correct={final_correct}")
    now = time.time()
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
    logger.info(
        f"[HLE_TRIAL_COMPLETE] ts_iso={ts_iso} ts_unix={now:.3f} task_id={e['id']} job_id={program_id} "
        f"status={task_status} correct={final_correct}",
    )
    clear_task_context()
    _release_program()
    return return_dict

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--max_rounds', type=int, default=50)
    parser.add_argument('--model_type', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--example_path', type=str)
    parser.add_argument('--concurrency', type=int, default=int(os.getenv("HLE_CONCURRENCY", "2")))
    parser.add_argument('--log_level', type=str, default=os.getenv("HLE_LOG_LEVEL", "INFO"))
    parser.add_argument('--log_file', type=str, default=os.getenv("HLE_LOG_FILE", ""))
    args = parser.parse_args()

    # global MODEL_NAME
    MODEL_NAME = args.model_name
    # global MODEL_TYPE
    MODEL_TYPE = args.model_type
    # global my_output_dir
    my_output_dir = args.output_dir
    # global MAX_ROUNDS
    MAX_ROUNDS = args.max_rounds
    if not os.path.isdir(os.path.join(my_output_dir,'answer_cache')):
        os.makedirs(os.path.join(my_output_dir,'answer_cache'))
    # global vllm_model_configs
    with open(args.model_config) as f:
        vllm_model_configs = json.load(f)

    # Configure structured profiling logs (file + optional stdout)
    log_file = (args.log_file or "").strip()
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except Exception:
            pass
    stream_flag = os.getenv("HLE_LOG_STREAM", "1").strip().lower() not in {"0", "false", "no"}
    configure_hle_logging(
        level=args.log_level,
        stream_handler=stream_flag,
        file_handler=log_file or None,
    )
    log_profile_event("run_start", model_name=args.model_name, concurrency=args.concurrency)

    with open(args.example_path) as f:
        lines = f.readlines()
    examples = []
    for eid, l in enumerate(lines):
        if not l.strip():
            continue
        raw_example = json.loads(l)
        raw_example['eid'] = eid
        examples.append([run_single,raw_example])

    # Never abort the whole run on a single task exception.
    tool_call_results = asyncio.run(run_all(examples, concurrency=args.concurrency, return_exceptions=True))


    
