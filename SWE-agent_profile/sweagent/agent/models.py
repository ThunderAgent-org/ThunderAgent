from __future__ import annotations
from vllm.reasoning.glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser
from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
import copy
import importlib
import importlib.util
import json
import os
import random
import shlex
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

import litellm
import litellm.types.utils
import requests
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, SecretStr
from swerex.exceptions import SwerexException
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent import REPO_ROOT
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FunctionCallingFormatError,
    InstanceCallLimitExceededError,
    InstanceCostLimitExceededError,
    ModelConfigurationError,
    TotalCostLimitExceededError,
)
from sweagent.tools.tools import ToolConfig
from sweagent.types import History, HistoryItem
from sweagent.utils.log import get_logger

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

try:  # vLLM is optional for offline local deployments
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None

_engine_exceptions_mod = importlib.util.find_spec("vllm.v1.engine.exceptions")
if _engine_exceptions_mod is not None:  # pragma: no cover - optional dependency
    EngineDeadError = getattr(importlib.import_module("vllm.v1.engine.exceptions"), "EngineDeadError", None)
else:  # pragma: no cover - optional dependency
    EngineDeadError = None

litellm.suppress_debug_info = True


_THREADS_THAT_USED_API_KEYS = []
"""Keeps track of thread orders so that we can choose the same API key for the same thread."""


def _history_to_openai_messages(
    history: History,
    *,
    convert_system_to_user: bool = False,
) -> list[dict[str, Any]]:
    """Convert internal history representation to OpenAI style messages."""

    history_copy = copy.deepcopy(history)
    messages: list[dict[str, Any]] = []

    for history_item in history_copy:
        role = history_item["role"]
        if role == "system" and convert_system_to_user:
            role = "user"

        if role == "tool":
            message: dict[str, Any] = {
                "role": role,
                "content": history_item["content"],
                "tool_call_id": history_item["tool_call_ids"][0],  # type: ignore[index]
            }
        elif (tool_calls := history_item.get("tool_calls")) is not None:
            message = {"role": role, "content": history_item["content"], "tool_calls": tool_calls}
            if thinking_blocks := history_item.get("thinking_blocks"):
                message["thinking_blocks"] = thinking_blocks
        else:
            message = {"role": role, "content": history_item["content"]}

        if "cache_control" in history_item:
            message["cache_control"] = history_item["cache_control"]

        messages.append(message)

    return messages


class RetryConfig(PydanticBaseModel):
    """This configuration object specifies how many times to retry a failed LM API call."""

    retries: int = 20
    """Number of retries"""
    min_wait: float = 10
    """Minimum wait time between retries (random exponential wait)"""
    max_wait: float = 120
    """Maximum wait time between retries (random exponential wait)"""


class GenericAPIModelConfig(PydanticBaseModel):
    """This configuration object specifies a LM like GPT4 or similar.
    The model will be served with the help of the `litellm` library.
    """

    name: str = Field(description="Name of the model.")

    per_instance_cost_limit: float = Field(
        default=3.0,
        description="Cost limit for every instance (task).",
    )
    total_cost_limit: float = Field(default=0.0, description="Total cost limit.")
    per_instance_call_limit: int = Field(default=0, description="Per instance call limit.")
    temperature: float = 0.0
    """Sampling temperature"""
    top_p: float | None = 1.0
    """Sampling top-p"""
    api_base: str | None = None
    api_version: str | None = None
    api_key: SecretStr | None = None
    """API key to the model. We recommend using environment variables to set this instead
    or putting your environment variables in a `.env` file.
    You can concatenate more than one key by separating them with `:::`, e.g.,
    `key1:::key2`.
    If field starts with `$`, it will be interpreted as an environment variable.
    """
    stop: list[str] = []
    """Custom stop sequences"""

    completion_kwargs: dict[str, Any] = {}
    """Additional kwargs to pass to `litellm.completion`"""

    convert_system_to_user: bool = False
    """Whether to convert system messages to user messages. This is useful for
    models that do not support system messages like o1.
    """

    retry: RetryConfig = RetryConfig()
    """Retry configuration: How often to retry after a failure (e.g., from a rate limit)
    etc.
    """

    delay: float = 0.0
    """Minimum delay before querying (this can help to avoid overusing the API if sharing
    it with other people).
    """

    fallbacks: list[dict[str, Any]] = []
    """List of fallbacks to try if the main model fails
    See https://docs.litellm.ai/docs/completion/reliable_completions#fallbacks-sdk
    for more information.
    """

    choose_api_key_by_thread: bool = True
    """Whether to choose the API key based on the thread name (if multiple are configured).
    This ensures that with
    run-batch, we use the same API key within a single-thread so that prompt caching still works.
    """

    max_input_tokens: int | None = None
    """If set, this will override the max input tokens for the model that we usually look
    up from `litellm.model_cost`.
    Use this for local models or if you want to set a custom max input token limit.
    If this value is exceeded, a `ContextWindowExceededError` will be raised.
    Set this to 0 to disable this check.
    """

    max_output_tokens: int | None = None
    """If set, this will override the max output tokens for the model that we usually look
    up from `litellm.model_cost`.
    Use this for local models or if you want to set a custom max output token limit.
    If this value is exceeded, a `ContextWindowExceededError` will be raised.
    Set this to 0 to disable this check.
    """

    litellm_model_registry: str | None = None
    """If set, this will override the default model registry for litellm.
    Use this for local models or models not (yet) in the default litellm model registry for tracking costs.
    """

    custom_tokenizer: dict[str, Any] | None = None
    """Override the default tokenizer for the model.
    Use the arguments of `litellm.create_pretrained_tokenizer`.
    Basic example: `{"identifier": "hf-internal-testing/llama-tokenizer"}`
    """

    # pydantic
    model_config = ConfigDict(extra="forbid")

    def get_api_keys(self) -> list[str]:
        """Returns a list of API keys that were explicitly set in this config.
        Does not return API keys that were set via environment variables/.env
        """
        if self.api_key is None:
            return []
        api_key = self.api_key.get_secret_value()
        if not api_key:
            return []
        if api_key.startswith("$"):
            env_var_name = api_key[1:]
            api_key = os.getenv(env_var_name, "")
            if not api_key:
                get_logger("swea-config", emoji="🔧").warning(f"Environment variable {env_var_name} not set")
                return []
        return api_key.split(":::")

    def choose_api_key(self) -> str | None:
        """Chooses an API key based on the API keys explicitly set in this config.
        If no API keys are set, returns None (which means that the API key will be
        taken from the environment variables/.env file).
        """
        api_keys = self.get_api_keys()
        if not api_keys:
            return None
        if not self.choose_api_key_by_thread:
            return random.choice(api_keys)
        thread_name = threading.current_thread().name
        if thread_name not in _THREADS_THAT_USED_API_KEYS:
            _THREADS_THAT_USED_API_KEYS.append(thread_name)
        thread_idx = _THREADS_THAT_USED_API_KEYS.index(thread_name)
        key_idx = thread_idx % len(api_keys)
        get_logger("config", emoji="🔧").debug(
            f"Choosing API key {key_idx} for thread {thread_name} (idx {thread_idx})"
        )
        return api_keys[key_idx]

    @property
    def id(self) -> str:
        name = self.name.replace("/", "--")
        if self.top_p is not None:
            top_p = f"{self.top_p:.2f}"
        else:
            top_p = "None"
        temperature = f"{self.temperature:.2f}"
        per_instance_cost_limit = f"{self.per_instance_cost_limit:.2f}"
        return f"{name}__t-{temperature}__p-{top_p}__c-{per_instance_cost_limit}"


class ReplayModelConfig(GenericAPIModelConfig):
    replay_path: Path = Field(description="Path to replay file when using the replay model.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )

    name: Literal["replay"] = Field(default="replay", description="Model name.")

    model_config = ConfigDict(extra="forbid")


class InstantEmptySubmitModelConfig(GenericAPIModelConfig):
    """Model that immediately submits an empty patch"""

    name: Literal["instant_empty_submit"] = Field(default="instant_empty_submit", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )
    delay: float = 0.0
    """Delay before answering"""

    model_config = ConfigDict(extra="forbid")


class LocalVLLMModelConfig(GenericAPIModelConfig):
    """Configuration for models served through a local vLLM OpenAI-compatible endpoint."""

    name: Literal["local_vllm"] = Field(default="local_vllm", description="Model name.")
    served_model_name: str = Field(
        description=(
            "Model identifier exposed by the vLLM server (passed as the `model` field in OpenAI requests)."
        )
    )
    api_base: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="Base URL for the vLLM OpenAI-compatible server.",
    )
    request_timeout: float = Field(
        default=120.0,
        description="Timeout (in seconds) for HTTP requests to the vLLM server.",
    )

    per_instance_cost_limit: float = Field(
        default=0.0,
        description="Cost limit per instance. vLLM responses do not provide cost information so this must be 0.",
    )
    total_cost_limit: float = Field(
        default=0.0,
        description="Total cost limit. vLLM responses do not provide cost information so this must be 0.",
    )

    model_config = ConfigDict(extra="forbid")


class LocalVLLMOfflineModelConfig(GenericAPIModelConfig):
    """Configuration for directly loading a vLLM engine without running an HTTP server."""

    name: Literal["local_vllm_offline"] = Field(default="local_vllm_offline", description="Model name.")
    model_path: Path = Field(description="Path or identifier of the model to load with vLLM.")
    tokenizer_path: Path | None = Field(default=None, description="Optional tokenizer path override.")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel degree for vLLM.")
    max_model_len: int | None = Field(default=None, description="Override model context length.")
    dtype: str | None = Field(default=None, description="Computation dtype passed to vLLM.")
    gpu_memory_utilization: float | None = Field(
        default=None,
        description="Optional GPU memory utilization hint passed to vLLM.",
    )
    swap_space: int | None = Field(default=None, description="CPU swap space (in GB) for vLLM.")
    enforce_eager: bool = Field(default=False, description="Enable eager execution in vLLM.")
    kv_cache_dtype: str | None = Field(default=None, description="KV cache dtype override.")
    enable_prefix_caching: bool = Field(
        default=True,
        description="Whether to allow vLLM to reuse kv-cache prefixes across turns.",
    )
    per_instance_cost_limit: float = Field(
        default=0.0,
        description="Cost limit per instance (must remain 0 for offline models).",
    )
    total_cost_limit: float = Field(
        default=0.0,
        description="Total cost limit (must remain 0 for offline models).",
    )

    model_config = ConfigDict(extra="forbid")
    
    def model_post_init(self, __context) -> None:
        """Validate configuration after model creation"""
        if self.per_instance_cost_limit != 0.0:
            raise ValueError("per_instance_cost_limit must be 0.0 for offline models")
        if self.total_cost_limit != 0.0:
            raise ValueError("total_cost_limit must be 0.0 for offline models")
        if self.gpu_memory_utilization is not None and (self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1):
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")


class HumanModelConfig(GenericAPIModelConfig):
    name: Literal["human"] = Field(default="human", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(default=0.0, description="Cost limit for all instances (tasks).")
    cost_per_call: float = 0.0
    catch_eof: bool = True
    """Whether to catch EOF and return 'exit' when ^D is pressed. Set to False when used in human_step_in mode."""
    model_config = ConfigDict(extra="forbid")


class HumanThoughtModelConfig(HumanModelConfig):
    name: Literal["human_thought"] = Field(default="human_thought", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )
    cost_per_call: float = 0.0

    model_config = ConfigDict(extra="forbid")


ModelConfig = Annotated[
    GenericAPIModelConfig
    | ReplayModelConfig
    | InstantEmptySubmitModelConfig
    | LocalVLLMModelConfig
    | LocalVLLMOfflineModelConfig
    | HumanModelConfig
    | HumanThoughtModelConfig,
    Field(union_mode="left_to_right"),
]


class GlobalStats(PydanticBaseModel):
    """This class tracks usage numbers (costs etc.) across all instances."""

    total_cost: float = 0
    """Cumulative cost for all instances so far"""

    last_query_timestamp: float = 0
    """Timestamp of the last query. Currently only used with API models."""


GLOBAL_STATS = GlobalStats()
"""This object tracks usage numbers (costs etc.) across all instances.
Please use the `GLOBAL_STATS_LOCK` lock when accessing this object to avoid race conditions.
"""

GLOBAL_STATS_LOCK = Lock()
"""Lock for accessing `GLOBAL_STATS` without race conditions"""


class InstanceStats(PydanticBaseModel):
    """This object tracks usage numbers (costs etc.) for a single instance."""

    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) + getattr(other, field) for field in self.model_fields.keys()},
        )

    def __sub__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) - getattr(other, field) for field in self.model_fields.keys()},
        )


class AbstractModel(ABC):
    def __init__(self, config: ModelConfig, tools: ToolConfig):
        self.config: ModelConfig
        self.stats: InstanceStats

    def reset_stats(self):
        self.stats = InstanceStats()

    @abstractmethod
    def query(self, history: History, action_prompt: str = "> ") -> dict: ...

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 if there is no limit."""
        return 0


def _handle_raise_commands(action: str) -> None:
    if action == "raise_runtime":
        raise SwerexException()
    elif action == "raise_cost":
        raise CostLimitExceededError()
    elif action == "raise_context":
        raise ContextWindowExceededError()
    elif action.startswith("raise_function_calling"):
        parts = shlex.split(action)
        error_code = parts[1]
        if len(parts) == 3:
            error_message = parts[2]
        assert len(parts) < 4
        raise FunctionCallingFormatError(error_message, error_code)  # type: ignore


class HumanModel(AbstractModel):
    def __init__(self, config: HumanModelConfig, tools: ToolConfig):
        """Model that allows for human-in-the-loop"""
        self.logger = get_logger("swea-lm", emoji="🤖")
        self.config: HumanModelConfig = config
        self.stats = InstanceStats()

        # Determine which commands require multi-line input
        self.multi_line_command_endings = {
            command.name: command.end_name for command in tools.commands if command.end_name is not None
        }
        self._readline_histfile = REPO_ROOT / ".swe-agent-human-history"
        self._load_readline_history()

    def _load_readline_history(self) -> None:
        """Load autocomplete history from file"""
        if readline is None:
            return
        if self._readline_histfile.is_file():
            self.logger.debug(f"Loading readline history from {self._readline_histfile}")
            readline.read_history_file(self._readline_histfile)

    def _save_readline_history(self) -> None:
        """Save autocomplete history to file"""
        if readline is None:
            return
        readline.write_history_file(self._readline_histfile)

    def _update_stats(
        self,
    ) -> None:
        self.stats.instance_cost += self.config.cost_per_call
        self.stats.api_calls += 1
        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            msg = f"Instance cost limit exceeded: {self.stats.instance_cost} > {self.config.per_instance_cost_limit}"
            raise InstanceCostLimitExceededError(msg)
        if 0 < self.config.total_cost_limit < self.stats.instance_cost:
            msg = f"Total cost limit exceeded: {self.stats.instance_cost} > {self.config.total_cost_limit}"
            raise TotalCostLimitExceededError(msg)

    def _query(
        self,
        history: History,
        action_prompt: str = "> ",
    ) -> dict:
        """Logic for handling user input to pass to SWEEnv"""
        action = input(action_prompt)
        self._save_readline_history()
        command_name = action.split()[0] if action.strip() else ""

        # Special handling for multi-line input actions (i.e. edit)
        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:
                    # Continue reading input until terminating keyword inputted
                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":  # do arbitrary multi-line input
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        else:
            # Input has escaped things like \n, so we need to unescape it
            action = action.encode("utf8").decode("unicode_escape")
        if action.strip() and action.strip().split()[0] == "spend_money":
            money = float(action.strip().split()[1])
            self.stats.instance_cost += money
            action = f"echo 'Spent {money} dollars'"
        _handle_raise_commands(action)
        self._update_stats()
        return {"message": action}

    def query(self, history: History, action_prompt: str = "> ", n: int | None = None, **kwargs) -> dict | list[dict]:
        """Wrapper to separate action prompt from formatting"""
        out = []
        n_samples = n or 1
        for _ in range(n_samples):
            try:
                out.append(self._query(history, action_prompt))
            except KeyboardInterrupt:
                print("^C (exit with ^D)")
                out.append(self.query(history, action_prompt))
            except EOFError:
                if self.config.catch_eof:
                    print("\nGoodbye!")
                    out.append({"message": "exit"})
                else:
                    # Re-raise EOFError when catch_eof is disabled
                    raise
        if n is None:
            return out[0]
        return out


class HumanThoughtModel(HumanModel):
    def query(self, history: History, **kwargs) -> dict:
        """Logic for handling user input (both thought + action) to pass to SWEEnv"""
        thought_all = ""
        thought = input("Thought (end w/ END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super()._query(history, action_prompt="Action: ")["message"]

        return {"message": f"{thought_all}\n```\n{action}\n```"}


class ReplayModel(AbstractModel):
    def __init__(self, config: ReplayModelConfig, tools: ToolConfig):
        """Model used for replaying a trajectory (i.e., taking all the actions for the `.traj` file
        and re-issuing them.
        """
        self.config = config
        self.stats = InstanceStats()

        if not self.config.replay_path.exists():
            msg = f"Replay file {self.config.replay_path} not found"
            raise FileNotFoundError(msg)

        self._replays = [
            list(json.loads(x).values())[0] for x in Path(self.config.replay_path).read_text().splitlines(keepends=True)
        ]
        self._replay_idx = 0
        self._action_idx = 0
        self.use_function_calling = tools.use_function_calling
        self.submit_command = tools.submit_command
        self.logger = get_logger("swea-lm", emoji="🤖")

    def _next_replay(self) -> None:
        """Called after last action"""
        self._replay_idx += 1
        self._action_idx = 0

    def query(self, history: History) -> dict:
        """Logic for tracking which replay action to pass to SWEEnv"""
        self.stats.api_calls += 1
        actions = self._replays[self._replay_idx]
        try:
            action = actions[self._action_idx]
        except IndexError:
            # log error
            self.logger.error("Reached end of replay trajectory without submitting. Submitting now.")
            if self.use_function_calling:
                action = {
                    "message": f"Calling `{self.submit_command}` to submit.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_submit",
                            "function": {
                                "name": self.submit_command,
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            else:
                action = f"```\n{self.submit_command}\n```"

        self._action_idx += 1

        # Assuming `submit` is always last action of replay trajectory
        if isinstance(action, str) and action == "submit":
            self._next_replay()
            return {"message": action}

        # Handle both dict and string actions
        if isinstance(action, dict):
            return action
        return {"message": action}


class PredeterminedTestModel(AbstractModel):
    def __init__(self, outputs: list[dict | str]):
        """Model that outputs a predetermined sequence of messages. Useful for testing."""
        self._outputs = outputs
        self._idx = -1
        self.stats = InstanceStats()

    def query(self, *args, **kwargs) -> dict:
        self._idx += 1
        output = self._outputs[self._idx]
        if isinstance(output, str):
            _handle_raise_commands(output)
            return {"message": output}
        if not isinstance(output, dict):
            msg = f"Output must be string or dict, got {type(output)}"
            raise ValueError(msg)
        result = {"message": output["message"]}
        if "tool_calls" in output:
            result["tool_calls"] = output["tool_calls"]
        return result


class InstantEmptySubmitTestModel(AbstractModel):
    def __init__(self, args: InstantEmptySubmitModelConfig, tools: ToolConfig):
        """This model immediately submits. Useful for testing purposes"""
        super().__init__(args, tools)
        self.config: InstantEmptySubmitModelConfig = args
        self.stats = InstanceStats()
        self._action_idx = 0

    def query(self, history: list[dict[str, str]]) -> dict:
        time.sleep(random.uniform(0, self.config.delay))
        # Need to at least do _something_ to submit
        if self._action_idx == 0:
            self._action_idx = 1
            action = (
                "DISCUSSION\n"
                "Let's reproduce the bug by creating a `reproduce.py` file.\n\n"
                "```\n"
                "touch reproduce.py\n"
                "```\n"
            )
        elif self._action_idx == 1:
            self._action_idx = 0
            action = "DISCUSSION\nThe task should be resolved, so let's submit the patch.\n\n```\nsubmit\n```\n"
        self.stats.api_calls += 1
        return {"message": action}


class LiteLLMModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        """Model served by the `litellm` library."""
        # Always copy config to avoid shared state between different instances
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if tools.use_function_calling:
            if not litellm.utils.supports_function_calling(model=self.config.name):
                msg = (
                    f"Model {self.config.name} does not support function calling. If your model"
                    " does not support function calling, you can use `parse_function='thought_action'` instead. "
                    "See https://swe-agent.com/latest/faq/ for more information."
                )
                self.logger.warning(msg)
        if self.config.litellm_model_registry is not None:
            with open(self.config.litellm_model_registry) as f:
                model_costs = json.load(f)
                litellm.register_model(model_costs)
        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            self.model_max_input_tokens = litellm.model_cost.get(self.config.name, {}).get("max_input_tokens")

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            self.model_max_output_tokens = litellm.model_cost.get(self.config.name, {}).get("max_output_tokens")
            # Special handling for Claude 3.7 models to set 64k context by default when beta header not present
            # See https://github.com/SWE-agent/SWE-agent/pull/1016
            is_claude_3_7 = "claude-3-7-sonnet" in self.config.name or "claude-sonnet-4" in self.config.name
            has_128k_beta_header = (
                self.config.completion_kwargs.get("extra_headers", {}).get("anthropic-beta") == "output-128k-2025-02-19"
            )
            if is_claude_3_7 and not has_128k_beta_header:
                self.model_max_output_tokens = 64000
                self.logger.warning(
                    "Claude 3.7/4 models do not support 128k context by default. "
                    "Setting max output tokens to 64k. To enable 128k context, please set the "
                    "completion_kwargs to {'extra_headers': {'anthropic-beta': 'output-128k-2025-02-19'}}."
                )

        self.lm_provider = litellm.model_cost.get(self.config.name, {}).get("litellm_provider", self.config.name)
        self.custom_tokenizer = None
        if self.config.custom_tokenizer is not None:
            self.custom_tokenizer = litellm.utils.create_pretrained_tokenizer(**self.config.custom_tokenizer)

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 if there is no limit."""
        return self.config.per_instance_cost_limit

    def _update_stats(self, *, input_tokens: int, output_tokens: int, cost: float) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. err
        self.logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        self.logger.debug(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={GLOBAL_STATS.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.config.total_cost_limit < GLOBAL_STATS.total_cost:
            self.logger.warning(f"Cost {GLOBAL_STATS.total_cost:.2f} exceeds limit {self.config.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise TotalCostLimitExceededError(msg)

        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            self.logger.warning(
                f"Cost {self.stats.instance_cost:.2f} exceeds limit {self.config.per_instance_cost_limit:.2f}"
            )
            msg = "Instance cost limit exceeded"
            raise InstanceCostLimitExceededError(msg)

        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            self.logger.warning(f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}")
            msg = "Per instance call limit exceeded"
            raise InstanceCallLimitExceededError(msg)

    def _sleep(self) -> None:
        elapsed_time = time.time() - GLOBAL_STATS.last_query_timestamp
        if elapsed_time < self.config.delay:
            time.sleep(self.config.delay - elapsed_time)
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.last_query_timestamp = time.time()

    def _single_query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        self._sleep()
        # Workaround for litellm bug https://github.com/SWE-agent/SWE-agent/issues/1109
        messages_no_cache_control = copy.deepcopy(messages)
        for message in messages_no_cache_control:
            if "cache_control" in message:
                del message["cache_control"]
            if "thinking_blocks" in message:
                del message["thinking_blocks"]
        input_tokens: int = litellm.utils.token_counter(
            messages=messages_no_cache_control,
            model=self.custom_tokenizer["identifier"] if self.custom_tokenizer is not None else self.config.name,
            custom_tokenizer=self.custom_tokenizer,
        )
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.config.api_base:
            # Not assigned a default value in litellm, so only pass this if it's set
            extra_args["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools
        # We need to always set max_tokens for anthropic models
        completion_kwargs = self.config.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens
        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
                model=self.config.name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                **completion_kwargs,
                **extra_args,
                n=n,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        self.logger.debug(f"Response: {response}")
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.name)
        except Exception as e:
            self.logger.debug(f"Error calculating cost: {e}, setting cost to 0.")
            if self.config.per_instance_cost_limit > 0 or self.config.total_cost_limit > 0:
                msg = (
                    f"Error calculating cost: {e} for your model {self.config.name}. If this is ok "
                    "(local models, etc.), please make sure you set `per_instance_cost_limit` and "
                    "`total_cost_limit` to 0 to disable this safety check."
                )
                self.logger.error(msg)
                raise ModelConfigurationError(msg)
            cost = 0
        choices: litellm.types.utils.Choices = response.choices  # type: ignore
        n_choices = n if n is not None else 1
        outputs = []
        output_tokens = 0
        for i in range(n_choices):
            output = choices[i].message.content or ""
            output_tokens += litellm.utils.token_counter(
                text=output,
                model=self.custom_tokenizer["identifier"] if self.custom_tokenizer is not None else self.config.name,
                custom_tokenizer=self.custom_tokenizer,
            )
            output_dict = {"message": output}
            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:  # type: ignore
                    tool_calls = [call.to_dict() for call in response.choices[i].message.tool_calls]  # type: ignore
                else:
                    tool_calls = []
                output_dict["tool_calls"] = tool_calls
            if (
                hasattr(response.choices[i].message, "thinking_blocks")  # type: ignore
                and response.choices[i].message.thinking_blocks  # type: ignore
            ):
                output_dict["thinking_blocks"] = response.choices[i].message.thinking_blocks  # type: ignore
            outputs.append(output_dict)
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return outputs

    def _query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        if n is None:
            return self._single_query(messages, temperature=temperature)
        outputs = []
        # not needed for openai, but oh well.
        for _ in range(n):
            outputs.extend(self._single_query(messages))
        return outputs

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception_info = ""
            if attempt.retry_state.outcome is not None and attempt.retry_state.outcome.exception() is not None:
                exception = attempt.retry_state.outcome.exception()
                exception_info = f" due to {exception.__class__.__name__}: {str(exception)}"

            self.logger.warning(
                f"Retrying LM query: attempt {attempt.retry_state.attempt_number} "
                f"(slept for {attempt.retry_state.idle_for:.2f}s)"
                f"{exception_info}"
            )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(min=self.config.retry.min_wait, max=self.config.retry.max_wait),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    ContextWindowExceededError,
                    CostLimitExceededError,
                    RuntimeError,
                    litellm.exceptions.UnsupportedParamsError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.PermissionDeniedError,
                    litellm.exceptions.ContextWindowExceededError,
                    litellm.exceptions.APIError,
                    litellm.exceptions.ContentPolicyViolationError,
                    TypeError,
                    litellm.exceptions.AuthenticationError,
                    ContentPolicyViolationError,
                    ModelConfigurationError,
                    KeyboardInterrupt,
                    IndexError,
                )
            ),
            before_sleep=retry_warning,
        ):
            with attempt:
                result = self._query(messages, n=n, temperature=temperature)
        if n is None or n == 1:
            return result[0]
        return result

    def _history_to_messages(
        self,
        history: History,
    ) -> list[dict[str, str]]:
        messages = _history_to_openai_messages(
            history,
            convert_system_to_user=self.config.convert_system_to_user,
        )
        n_cache_control = str(messages).count("cache_control")
        self.logger.debug(f"n_cache_control: {n_cache_control}")
        return messages


class LocalVLLMModel(AbstractModel):
    def __init__(self, config: LocalVLLMModelConfig, tools: ToolConfig):
        self.config = config
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if self.config.per_instance_cost_limit not in (None, 0) or self.config.total_cost_limit not in (None, 0):
            self.logger.warning(
                "Cost tracking is not available for local vLLM deployments. "
                "Set both per_instance_cost_limit and total_cost_limit to 0 to silence this warning."
            )

    def _prepare_messages(self, history: History) -> list[dict[str, Any]]:
        messages = _history_to_openai_messages(
            history,
            convert_system_to_user=self.config.convert_system_to_user,
        )
        # Remove OpenAI unsupported metadata fields
        for message in messages:
            message.pop("cache_control", None)
            message.pop("thinking_blocks", None)
        return messages

    def _send_request(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}

        api_key = self.config.choose_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": self.config.served_model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "n": n,
        }

        if self.config.max_output_tokens:
            payload["max_tokens"] = self.config.max_output_tokens

        if self.tools.use_function_calling:
            payload["tools"] = self.tools.tools

        if self.config.completion_kwargs:
            payload.update(self.config.completion_kwargs)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.request_timeout)
        except requests.Timeout as exc:
            msg = f"Request to vLLM server timed out after {self.config.request_timeout}s"
            raise ModelConfigurationError(msg) from exc
        except requests.RequestException as exc:
            msg = f"Failed to reach vLLM server at {self.config.api_base}: {exc}"
            raise ModelConfigurationError(msg) from exc

        if response.status_code != 200:
            detail: str
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            if response.status_code == 400 and "context" in detail.lower():
                raise ContextWindowExceededError(detail)
            raise ModelConfigurationError(f"vLLM request failed ({response.status_code}): {detail}")

        data = response.json()
        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise ModelConfigurationError("vLLM response did not contain any choices")

        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        self.stats.tokens_sent += prompt_tokens
        self.stats.tokens_received += completion_tokens
        self.stats.api_calls += 1

        outputs: list[dict[str, Any]] = []
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content") or ""
            output: dict[str, Any] = {"message": content}
            if tool_calls := message.get("tool_calls"):
                output["tool_calls"] = tool_calls
            if thinking_blocks := message.get("thinking_blocks"):
                output["thinking_blocks"] = thinking_blocks
            outputs.append(output)

        return outputs

    def query(
        self,
        history: History,
        n: int = 1,
        temperature: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        messages = self._prepare_messages(history)
        outputs = self._send_request(messages, n=n, temperature=temperature)
        if n == 1:
            return outputs[0]
        return outputs

    @property
    def instance_cost_limit(self) -> float:
        return 0.0


class LocalVLLMOfflineModel(AbstractModel):
    _ENGINE_CACHE: dict[str, Any] = {}
    _ENGINE_LOADING: dict[str, threading.Event] = {}
    _CACHE_LOCK = threading.RLock()

    def __init__(self, config: LocalVLLMOfflineModelConfig, tools: ToolConfig):
        if LLM is None or SamplingParams is None:
            msg = "vLLM is not installed. Please `pip install vllm` to use local_vllm_offline."
            raise ModelConfigurationError(msg)

        self.config = config
        self.tools = tools
        self.stats = InstanceStats()
        self.logger = get_logger("swea-lm", emoji="🤖")

        if tools.use_function_calling:
            self.logger.info(
                "Function calling enabled for local_vllm_offline. "
                "Will attempt to use native vLLM function calling if supported by the model."
            )
            # Note: We can't check function calling support without loading the model,
            # but we'll log a warning if tools parameter is not accepted
        if self.config.per_instance_cost_limit not in (None, 0) or self.config.total_cost_limit not in (None, 0):
            self.logger.warning(
                "Cost tracking is not available for offline vLLM deployments. "
                "Set both per_instance_cost_limit and total_cost_limit to 0 to silence this warning."
            )

        llm_kwargs: dict[str, Any] = {
            "model": str(self.config.model_path),
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }
        if self.config.tokenizer_path is not None:
            llm_kwargs["tokenizer"] = str(self.config.tokenizer_path)
        if self.config.dtype is not None:
            llm_kwargs["dtype"] = self.config.dtype
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len
        if self.config.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = self.config.gpu_memory_utilization
        if self.config.swap_space is not None:
            llm_kwargs["swap_space"] = self.config.swap_space
        if self.config.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if self.config.kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = self.config.kv_cache_dtype

        self._llm_kwargs = llm_kwargs.copy()
        self._engine_key, self.llm, created = self._get_or_create_engine(self._llm_kwargs)
        if created:
            self.logger.info(
                "Loaded vLLM engine for %s (tensor_parallel=%s)",
                llm_kwargs.get("model", "<unknown>"),
                llm_kwargs.get("tensor_parallel_size"),
            )
        else:
            self.logger.debug(
                "Reusing cached vLLM engine for %s", llm_kwargs.get("model", "<unknown>")
            )

    def _prepare_messages(self, history: History) -> list[dict[str, Any]]:
        messages = _history_to_openai_messages(
            history,
            convert_system_to_user=self.config.convert_system_to_user,
        )
        
        return messages

    @classmethod
    def _engine_cache_key(cls, llm_kwargs: dict[str, Any]) -> str:
        normalized: dict[str, Any] = {}
        for key, value in llm_kwargs.items():
            if isinstance(value, Path):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        try:
            return json.dumps(normalized, sort_keys=True)
        except TypeError:
            # Fallback to repr when encountering non-serializable values
            return repr(sorted(normalized.items()))

    @classmethod
    def _drop_engine(cls, key: str) -> None:
        with cls._CACHE_LOCK:
            engine = cls._ENGINE_CACHE.pop(key, None)
            if engine is not None:
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        pass
            cls._ENGINE_LOADING.pop(key, None)

    @classmethod
    def _get_or_create_engine(cls, llm_kwargs: dict[str, Any], *, force_reload: bool = False) -> tuple[str, Any, bool]:
        key = cls._engine_cache_key(llm_kwargs)

        if force_reload:
            cls._drop_engine(key)

        with cls._CACHE_LOCK:
            cached = cls._ENGINE_CACHE.get(key)
            if cached is not None:
                return key, cached, False

            event = cls._ENGINE_LOADING.get(key)
            if event is None:
                event = threading.Event()
                cls._ENGINE_LOADING[key] = event
                creator = True
            else:
                creator = False

        if not creator:
            event.wait()
            with cls._CACHE_LOCK:
                cached_after_wait = cls._ENGINE_CACHE.get(key)
                if cached_after_wait is None:
                    raise ModelConfigurationError("vLLM engine initialisation failed in another thread.")
                return key, cached_after_wait, False

        try:
            engine = LLM(**llm_kwargs)
        except Exception as exc:  # pragma: no cover - passthrough to user
            with cls._CACHE_LOCK:
                cls._ENGINE_LOADING.pop(key, None)
                event.set()
            msg = f"Failed to initialise vLLM engine with kwargs {llm_kwargs}: {exc}"
            raise ModelConfigurationError(msg) from exc

        with cls._CACHE_LOCK:
            cls._ENGINE_CACHE[key] = engine
            cls._ENGINE_LOADING.pop(key, None)
            event.set()

        return key, engine, True

    def _refresh_engine(self) -> None:
        self.logger.warning("Refreshing cached vLLM engine after failure; reinitialising engine core.")
        self.__class__._drop_engine(self._engine_key)
        self._engine_key, self.llm, _ = self.__class__._get_or_create_engine(
            self._llm_kwargs,
            force_reload=True,
        )

    @classmethod
    def clear_engine_cache(cls) -> None:
        with cls._CACHE_LOCK:
            keys = list(cls._ENGINE_CACHE.keys())
        for key in keys:
            cls._drop_engine(key)
    
    def _get_chat_template_and_tools(self) -> tuple[str | None, list[dict] | None]:
        """Get chat template and tools for vLLM engine"""
        # For function calling, we want vLLM to handle tools natively if possible
        if self.tools.use_function_calling and self.tools.tools:
            return None, self.tools.tools  # Let vLLM handle tools natively
        return None, None
    
    def _extract_tool_calls_from_output(self, messages: Any, output_obj: Any) -> tuple[list[dict[str, Any]] | None, Any]:
        """Extract tool calls and thinking blocks from vLLM output object if available"""
        def extract_clean_thought(thought_text: str) -> str:
            if not thought_text:
                return ""
            
            tool_call_start = thought_text.find('<tool_call>')
            
            if tool_call_start != -1:
                clean_thought = thought_text[:tool_call_start].strip()
            else:
                clean_thought = thought_text.strip()
            return clean_thought
        tool_calls = None
        thinking_blocks = None
        request = ChatCompletionRequest(
                                model="/data/models/GLM-4.5-FP8",
                                messages=messages,
                                tools=self.tools.tools
                            )
        toolparser = Glm4MoeModelToolParser(tokenizer=self.llm.get_tokenizer())
        thinkparser = Glm4MoeModelReasoningParser(tokenizer=self.llm.get_tokenizer())

        parsed = toolparser.extract_tool_calls(output_obj.text, request=request)
        raw_tool_calls = [tool.model_dump() for tool in parsed.tool_calls]

        if raw_tool_calls:
            dedup: dict[tuple[str | None, str | None], dict[str, Any]] = {}
            for tool_call in reversed(raw_tool_calls):
                function_dict = tool_call.get("function", {}) or {}
                key = (
                    function_dict.get("name"),
                    json.dumps(function_dict.get("arguments"), sort_keys=True)
                    if isinstance(function_dict.get("arguments"), (dict, list))
                    else function_dict.get("arguments"),
                )
                dedup[key] = tool_call
            tool_calls = list(reversed(list(dedup.values())))
        else:
            tool_calls = None
        thinking_blocks, thought = thinkparser.extract_reasoning_content(output_obj.text, request=request)
        cleanthought = extract_clean_thought(thought)
        return tool_calls, thinking_blocks, cleanthought

    def _sampling_params(self, *, n: int, temperature: float | None) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "n": n,
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.max_output_tokens is not None and self.config.max_output_tokens > 0:
            kwargs["max_tokens"] = self.config.max_output_tokens
        if self.config.stop:
            kwargs["stop"] = self.config.stop
        
        # Add additional sampling parameters from completion_kwargs
        if self.config.completion_kwargs:
            for key, value in self.config.completion_kwargs.items():
                if key in ["top_k", "repetition_penalty", "length_penalty", "presence_penalty", "frequency_penalty"]:
                    kwargs[key] = value
        
        return SamplingParams(**kwargs)

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                text_chunks: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_chunks.append(block.get("text", ""))
                content = "".join(text_chunks)
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _execute_chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: Any,
    ) -> list[Any]:
        def _chat_call() -> Any:
            if self.tools.use_function_calling and self.tools.tools:
                return self.llm.chat(messages, tools=self.tools.tools, sampling_params=SamplingParams(temperature=0, max_tokens=2048))
            return self.llm.chat(messages=messages, sampling_params=SamplingParams(temperature=0, max_tokens=2048))

        try:
            outputs = _chat_call()
        except Exception as exc:
            if EngineDeadError is not None and isinstance(exc, EngineDeadError):
                self.logger.warning("vLLM engine core terminated unexpectedly; attempting one reload.")
                self._refresh_engine()
                try:
                    outputs = _chat_call()
                except Exception as exc_retry:
                    msg_retry = f"Failed to execute vLLM generation after engine reload: {exc_retry}"
                    raise ModelConfigurationError(msg_retry) from exc_retry
            else:
                msg = f"Failed to execute vLLM generation: {exc}"
                raise ModelConfigurationError(msg) from exc
        return outputs if isinstance(outputs, list) else [outputs]

    def query(
        self,
        history: History,
        n: int = 1,
        temperature: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        messages = self._prepare_messages(history)
        sampling_params = self._sampling_params(n=n, temperature=temperature)
        request_outputs = self._execute_chat(messages, sampling_params)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        outputs: list[dict[str, Any]] = []

        for request_output in request_outputs:
            # Get prompt tokens (should be consistent across all outputs)
            prompt_token_ids = getattr(request_output, "prompt_token_ids", []) or []
            if len(prompt_token_ids) > total_prompt_tokens:
                total_prompt_tokens = len(prompt_token_ids)
            
            # Process generated outputs
            generated = getattr(request_output, "outputs", [])
            for i, result in enumerate(generated):
                if i >= n:  # Respect the requested number of samples
                    break
                text = getattr(result, "text", "")
                
                # Try different attribute names for token IDs
                token_ids = (
                    getattr(result, "token_ids", None) or
                    getattr(result, "output_token_ids", None) or
                    []
                )
                total_completion_tokens += len(token_ids)
                
                # Handle tool calls and thinking blocks
                output_dict = {"message": text}
                if self.tools.use_function_calling:
                    # First check if vLLM returned tool calls natively
                    tool_calls, thinking_blocks, thought = self._extract_tool_calls_from_output(messages, result)
                    if tool_calls:
                        output_dict["tool_calls"] = tool_calls
                        self.logger.debug(f"Extracted {len(tool_calls)} tool call(s) from vLLM output")
                    else:
                        # No native tool calls, set empty list (model didn't call any tools)
                        output_dict["tool_calls"] = []
                else:
                    # Still check for thinking blocks even without function calling
                    _, thought = self._extract_tool_calls_from_output(result)
                
                # Add thinking blocks if available
                if thought:
                    output_dict["thought"] = thought
                if thinking_blocks:
                    output_dict["thinking_blocks"] = thinking_blocks
                outputs.append(output_dict)

        if not outputs:
            raise ModelConfigurationError("vLLM offline engine returned no outputs")

        self.stats.tokens_sent += total_prompt_tokens
        self.stats.tokens_received += total_completion_tokens
        self.stats.api_calls += 1

        # Enhanced logging similar to LiteLLMModel
        self.logger.debug(
            f"prompt_tokens={total_prompt_tokens:,}, "
            f"completion_tokens={total_completion_tokens:,}, "
            f"api_calls={self.stats.api_calls}, "
            f"outputs={len(outputs)}"
        )
        
        # Log individual outputs for debugging
        for i, output in enumerate(outputs):
            has_tools = "tool_calls" in output and output["tool_calls"]
            has_thinking = "thinking_blocks" in output
            self.logger.debug(
                f"Output {i+1}: message_len={len(output['message'])}, "
                f"has_tools={has_tools}, has_thinking={has_thinking}"
            )

        if n == 1:
            return outputs[0]
        return outputs

    @property
    def instance_cost_limit(self) -> float:
        return 0.0


class LocalVLLMModel(AbstractModel):
    def __init__(self, config: LocalVLLMModelConfig, tools: ToolConfig):
        self.config = config
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if self.config.per_instance_cost_limit not in (None, 0) or self.config.total_cost_limit not in (None, 0):
            self.logger.warning(
                "Cost tracking is not available for local vLLM deployments. "
                "Set both per_instance_cost_limit and total_cost_limit to 0 to silence this warning."
            )

    def _prepare_messages(self, history: History) -> list[dict[str, Any]]:
        messages = _history_to_openai_messages(
            history,
            convert_system_to_user=self.config.convert_system_to_user,
        )
        # Remove OpenAI unsupported metadata fields
        for message in messages:
            message.pop("cache_control", None)
            message.pop("thinking_blocks", None)
        return messages

    def _send_request(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}

        api_key = self.config.choose_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": self.config.served_model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "n": n,
        }

        if self.config.max_output_tokens:
            payload["max_tokens"] = self.config.max_output_tokens

        if self.tools.use_function_calling:
            payload["tools"] = self.tools.tools

        if self.config.completion_kwargs:
            payload.update(self.config.completion_kwargs)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.request_timeout)
        except requests.Timeout as exc:
            msg = f"Request to vLLM server timed out after {self.config.request_timeout}s"
            raise ModelConfigurationError(msg) from exc
        except requests.RequestException as exc:
            msg = f"Failed to reach vLLM server at {self.config.api_base}: {exc}"
            raise ModelConfigurationError(msg) from exc

        if response.status_code != 200:
            detail: str
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            if response.status_code == 400 and "context" in detail.lower():
                raise ContextWindowExceededError(detail)
            raise ModelConfigurationError(f"vLLM request failed ({response.status_code}): {detail}")

        data = response.json()
        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise ModelConfigurationError("vLLM response did not contain any choices")

        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        self.stats.tokens_sent += prompt_tokens
        self.stats.tokens_received += completion_tokens
        self.stats.api_calls += 1

        outputs: list[dict[str, Any]] = []
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content") or ""
            output: dict[str, Any] = {"message": content}
            if tool_calls := message.get("tool_calls"):
                output["tool_calls"] = tool_calls
            if thinking_blocks := message.get("thinking_blocks"):
                output["thinking_blocks"] = thinking_blocks
            outputs.append(output)

        return outputs

    def query(
        self,
        history: History,
        n: int = 1,
        temperature: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        messages = self._prepare_messages(history)
        outputs = self._send_request(messages, n=n, temperature=temperature)
        if n == 1:
            return outputs[0]
        return outputs

    @property
    def instance_cost_limit(self) -> float:
        return 0.0


def get_model(args: ModelConfig, tools: ToolConfig) -> AbstractModel:
    """Returns correct model object given arguments and commands"""
    # Convert GenericAPIModelConfig to specific model config if needed
    #logger = get_logger("swea-model", emoji="🔧")
    #logger.info(f"Model config: {args}")  # 使用日志而不是print
    if isinstance(args, GenericAPIModelConfig) and not isinstance(
        args, HumanModelConfig | HumanThoughtModelConfig | ReplayModelConfig | InstantEmptySubmitModelConfig
    ):
        if args.name == "human":
            args = HumanModelConfig(**args.model_dump())
        elif args.name == "human_thought":
            args = HumanThoughtModelConfig(**args.model_dump())
        elif args.name == "replay":
            args = ReplayModelConfig(**args.model_dump())
        elif args.name == "instant_empty_submit":
            args = InstantEmptySubmitModelConfig(**args.model_dump())

    if args.name == "human":
        assert isinstance(args, HumanModelConfig), f"Expected {HumanModelConfig}, got {args}"
        return HumanModel(args, tools)
    if args.name == "human_thought":
        assert isinstance(args, HumanThoughtModelConfig), f"Expected {HumanThoughtModelConfig}, got {args}"
        return HumanThoughtModel(args, tools)
    if args.name == "replay":
        assert isinstance(args, ReplayModelConfig), f"Expected {ReplayModelConfig}, got {args}"
        return ReplayModel(args, tools)
    elif args.name == "instant_empty_submit":
        assert isinstance(args, InstantEmptySubmitModelConfig), f"Expected {InstantEmptySubmitModelConfig}, got {args}"
        return InstantEmptySubmitTestModel(args, tools)
    elif args.name == "local_vllm":
        assert isinstance(args, LocalVLLMModelConfig), f"Expected {LocalVLLMModelConfig}, got {args}"
        return LocalVLLMModel(args, tools)
    elif args.name == "local_vllm_offline":
        assert isinstance(args, LocalVLLMOfflineModelConfig), f"Expected {LocalVLLMOfflineModelConfig}, got {args}"
        return LocalVLLMOfflineModel(args, tools)
    assert isinstance(args, GenericAPIModelConfig), f"Expected {GenericAPIModelConfig}, got {args}"
    return LiteLLMModel(args, tools)