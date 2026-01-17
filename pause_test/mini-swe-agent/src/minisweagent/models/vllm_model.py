import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from openai import OpenAI, BadRequestError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("vllm_model")


class ContextLengthExceededError(Exception):
    """Raised when the context length exceeds the model's maximum."""
    pass


@dataclass
class VllmModelConfig:
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    base_url: str = "http://localhost:8100/v1"
    api_key: str = "EMPTY"  # vLLM doesn't require API key
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    job_id: int = 0  # Instance/job ID for tracking
    step_limit: int = 0  # Step limit for calculating is_last_step
    stream: bool = True  # Enable streaming responses
    timeout: float | None = None  # Request timeout; None disables client-side timeout
    max_completion_tokens: int = 2048  # Maximum tokens to generate per response


class VllmModel:
    """Model class for vLLM inference server.

    This class provides an interface to vLLM (https://github.com/vllm-project/vllm),
    a fast and easy-to-use library for LLM inference and serving.

    vLLM exposes an OpenAI-compatible API, so we use the OpenAI client to communicate with it.

    Args:
        config_class: Configuration class to use (default: VllmModelConfig)
        **kwargs: Additional arguments to pass to the config class

    Example:
        >>> model = VllmModel(base_url="http://localhost:8000/v1", model_name="meta-llama/Llama-3.1-70B-Instruct")
        >>> response = model.query([{"role": "user", "content": "Hello!"}])
    """

    def __init__(self, *, config_class: type = VllmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0  # vLLM doesn't have cost, set to 0
        self.n_calls = 0

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _filter_openai_params(params: dict[str, Any]) -> dict[str, Any]:
        """Filter parameters to only include those compatible with OpenAI API."""
        # List of valid parameters for OpenAI chat completions API
        # Based on: https://platform.openai.com/docs/api-reference/chat/create
        valid_params = {
            "temperature",
            "top_p",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "seed",
            "tools",
            "tool_choice",
            "response_format",
            "logprobs",
            "top_logprobs",
        }
        filtered = {k: v for k, v in params.items() if k in valid_params}
        if filtered != params:
            dropped = set(params.keys()) - set(filtered.keys())
            logger.debug(f"Dropped incompatible parameters for OpenAI API: {dropped}")
        return filtered

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
                ContextLengthExceededError,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Internal query method with retry logic."""
        try:
            # Merge config model_kwargs and runtime kwargs, then filter for OpenAI compatibility
            all_params = self.config.model_kwargs | kwargs

            # Add stream parameter from config if not explicitly provided
            if "stream" not in all_params:
                all_params["stream"] = self.config.stream

            # Add max_completion_tokens from config if not explicitly provided
            if "max_completion_tokens" not in all_params and "max_tokens" not in all_params:
                all_params["max_completion_tokens"] = self.config.max_completion_tokens

            filtered_params = self._filter_openai_params(all_params)

            # Calculate is_last_step based on step_limit
            is_last_step = False
            if self.config.step_limit > 0:
                is_last_step = (self.n_calls + 1) >= self.config.step_limit

            # Prepare extra_body with Continuum-specific parameters
            job_id_value = self.config.job_id if self.config.job_id > 0 else self.n_calls + 1
            extra_body = {
                "ignore_eos": False,
                "job_id": str(job_id_value),
                "is_last_step": is_last_step,
                "session_id": str(job_id_value),
            }

            return self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                extra_body=extra_body,
                **filtered_params
            )
        except BadRequestError as e:
            # Check if this is a context length exceeded error
            error_message = str(e)
            if "maximum context length" in error_message.lower() or "context length" in error_message.lower():
                logger.error(f"Context length exceeded: {error_message}")
                raise ContextLengthExceededError(error_message) from e
            logger.error(f"Bad request error querying vLLM server: {e}")
            raise
        except Exception as e:
            logger.error(f"Error querying vLLM server: {e}")
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the vLLM model with a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the completion API

        Returns:
            Dictionary with 'content' and 'extra' keys
        """
        response = self._query(messages, **kwargs)

        # Track calls but no cost for vLLM
        self.n_calls += 1
        cost = 0.0  # No cost for vLLM
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Check if streaming is enabled
        use_stream = kwargs.get("stream", self.config.stream)

        if use_stream:
            # Handle streaming response
            content_chunks = []
            full_response = None

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content_chunks.append(delta.content)
                # Keep the last chunk for metadata
                full_response = chunk

            content = "".join(content_chunks)

            return {
                "content": content,
                "extra": {
                    "response": full_response.model_dump() if full_response else {},
                    "streamed": True,
                },
            }
        else:
            # Handle non-streaming response
            return {
                "content": response.choices[0].message.content or "",
                "extra": {
                    "response": response.model_dump(),
                    "streamed": False,
                },
            }

    def query_stream_timed(self, messages: list[dict[str, str]], **kwargs) -> tuple[dict, dict[str, float]]:
        """Streaming query that also records first/last token timestamps.

        Returns:
            (response_dict, timing_dict)

        timing_dict contains:
            - first_token_ts: epoch seconds when first content token arrived
            - last_token_ts: epoch seconds when stream finished

        Note: The agent records query start time externally.
        """
        # Force streaming so we can observe first token latency.
        kwargs = dict(kwargs)
        kwargs["stream"] = True

        response = self._query(messages, **kwargs)

        # Track calls but no cost for vLLM
        self.n_calls += 1
        cost = 0.0
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        first_token_ts: float | None = None
        full_response = None
        content_chunks: list[str] = []

        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    content_chunks.append(delta.content)
            full_response = chunk

        last_token_ts = time.time()
        if first_token_ts is None:
            first_token_ts = last_token_ts

        content = "".join(content_chunks)
        return (
            {
                "content": content,
                "extra": {
                    "response": full_response.model_dump() if full_response else {},
                    "streamed": True,
                },
            },
            {
                "first_token_ts": first_token_ts,
                "last_token_ts": last_token_ts,
            },
        )

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for this model instance."""
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
