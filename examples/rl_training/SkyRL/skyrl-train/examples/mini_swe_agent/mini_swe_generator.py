import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj
from minisweagent.config import get_config_path
from .mini_swe_utils import evaluate_trajectory, get_sb_environment

from omegaconf import DictConfig
from skyrl_train.config import GeneratorConfig, SkyRLGymConfig
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


class DefaultAgentWithReminder(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_timings = []  # Per-step timing: [{step, model_call_time, tool_call_time}, ...]

    def query(self):
        """Override to time model calls (vLLM inference)."""
        start = time.time()
        result = super().query()
        model_time = time.time() - start
        self.step_timings.append({
            "step": self.model.n_calls,
            "model_call_time": model_time,
            "tool_call_time": None,
        })
        return result

    def get_observation(self, response: dict) -> dict:
        """Execute the action, time it, and return the output."""
        action = self.parse_action(response)

        # Extract command string from parsed action
        if isinstance(action, list):
            commands = [a.get("command", str(a)) if isinstance(a, dict) else str(a) for a in action]
        elif isinstance(action, dict):
            commands = [action.get("command", str(action))]
        else:
            commands = [str(action)]

        # Get container ID for full docker exec reconstruction
        container_id = getattr(getattr(self, "env", None), "container_id", None) or ""

        start = time.time()
        output = self.execute_action(action)
        tool_time = time.time() - start

        # Update the last timing entry with tool call time and command
        if self.step_timings:
            self.step_timings[-1]["tool_call_time"] = tool_time
            self.step_timings[-1]["command"] = commands[0] if len(commands) == 1 else commands
            if container_id:
                self.step_timings[-1]["docker_exec"] = f"docker exec {container_id} {commands[0]}"

        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: Union[GeneratorConfig, DictConfig],
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)

    # Inject program_id for ThunderAgent capacity scheduling.
    # Each trajectory is a unique "program" in ThunderAgent's model.
    # LiteLLM passes extra_body fields through to the request JSON body,
    # and ThunderAgent extracts program_id from extra_body.program_id.
    program_id = f"{instance['instance_id']}_{trajectory_id.repetition_id}"
    model_config.setdefault("model_kwargs", {}).setdefault("extra_body", {})["program_id"] = program_id

    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            save_traj(agent, path, exit_status=exit_status, result=result, extra_info=extra_info, reward=reward, eval_error=eval_error)  # type: ignore[arg-type]

    # Release program from ThunderAgent (free capacity tracking resources).
    # Fire-and-forget: don't block on failure (ThunderAgent may not be running).
    try:
        import httpx
        httpx.post(
            f"{os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8001/v1').rsplit('/v1', 1)[0]}/programs/release",
            json={"program_id": program_id},
            timeout=5.0,
        )
    except Exception:
        pass

    step_timings = agent.step_timings if agent is not None and hasattr(agent, "step_timings") else []

    # Persist step_timings (model_call_time, tool_call_time, command) to disk
    if step_timings:
        try:
            timings_dir = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
            timings_dir.mkdir(parents=True, exist_ok=True)
            timings_path = timings_dir / f"{instance['instance_id']}_{trajectory_id.repetition_id}_timings.json"
            with open(timings_path, "w") as f:
                json.dump(step_timings, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save step_timings: {e}")
    return (agent.messages if agent is not None else [], reward, error, step_timings)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: Union[GeneratorConfig, DictConfig],
        skyrl_gym_cfg: Union[SkyRLGymConfig, DictConfig],
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)

        self.http_server_inference_engine_client_host = generator_cfg.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.http_endpoint_port

        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    def _make_dummy_output(
        self,
        prompt: ConversationType,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[int], float, str, List[int], List[int], None]:
        """Create a valid dummy output for failed trajectories.

        Returns data that passes all downstream trainer checks but contributes
        zero to training (reward=0, loss_mask all zeros, single EOS token).
        """
        eos_token_id = self.tokenizer.eos_token_id
        dummy_response_ids = [eos_token_id]
        dummy_loss_mask = [0]
        dummy_reward = 0.0
        dummy_stop_reason = "error"

        # Construct prompt_ids from available messages
        try:
            if messages is not None and len(messages) >= 2:
                # Use actual messages (system + user) when available
                dummy_prompt_ids = self.tokenizer.apply_chat_template(
                    messages[:2], add_generation_prompt=False, tokenize=True
                )
            elif isinstance(prompt, list) and len(prompt) >= 2:
                # Fall back to the input prompt conversation
                dummy_prompt_ids = self.tokenizer.apply_chat_template(
                    prompt[:2], add_generation_prompt=False, tokenize=True
                )
            else:
                dummy_prompt_ids = [eos_token_id]
        except Exception:
            dummy_prompt_ids = [eos_token_id]

        return (dummy_response_ids, dummy_reward, dummy_stop_reason, dummy_loss_mask, dummy_prompt_ids, None)

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error, step_timings = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            from loguru import logger
            logger.warning(f"Empty messages for trajectory {trajectory_id}. Returning dummy output.")
            return self._make_dummy_output(prompt)

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            from loguru import logger
            logger.warning(
                "Found no assistant messages for this trajectory. Returning dummy output. "
                "Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
            return self._make_dummy_output(prompt, messages)
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def _collect_vllm_metrics(self, step: int, metrics_log: list, interval: float = 2.0):
        """Background task to collect vLLM metrics every `interval` seconds during rollout."""
        try:
            import httpx
        except ImportError:
            return
        base_url = self.base_url
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(f"{base_url}/metrics", timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        data["step"] = step
                        data["collected_at"] = time.time()
                        metrics_log.append(data)
                except Exception:
                    pass
                await asyncio.sleep(interval)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        # Start background vLLM metrics collection (every 2s during rollout)
        vllm_metrics_log: list = []
        metrics_task = asyncio.create_task(
            self._collect_vllm_metrics(batch_metadata.global_step, vllm_metrics_log)
        )

        from loguru import logger

        # Rate-limit Docker container launches: start trajectories in batches
        # to avoid overwhelming the Docker daemon with simultaneous container creation.
        ROLLOUT_BATCH_SIZE = 64
        ROLLOUT_BATCH_DELAY = 30.0  # seconds between batches

        all_tasks = []
        total = len(prompts)
        total_batches = (total + ROLLOUT_BATCH_SIZE - 1) // ROLLOUT_BATCH_SIZE

        for batch_start in range(0, total, ROLLOUT_BATCH_SIZE):
            batch_end = min(batch_start + ROLLOUT_BATCH_SIZE, total)
            batch_num = batch_start // ROLLOUT_BATCH_SIZE + 1

            if batch_start > 0:
                logger.info(
                    f"Rollout batch {batch_num}/{total_batches}: waiting {ROLLOUT_BATCH_DELAY}s before launching next batch..."
                )
                await asyncio.sleep(ROLLOUT_BATCH_DELAY)

            logger.info(
                f"Rollout batch {batch_num}/{total_batches}: launching {batch_end - batch_start} trajectories [{batch_start}:{batch_end}]"
            )
            for i in range(batch_start, batch_end):
                task = asyncio.create_task(
                    self.minisweagent_agent_loop(
                        prompts[i],
                        env_extras[i],
                        max_tokens=max_tokens,
                        max_input_length=max_input_length,
                        sampling_params=sampling_params,
                        trajectory_id=trajectory_ids[i],
                        batch_metadata=batch_metadata,
                    )
                )
                all_tasks.append(task)

        all_outputs = await asyncio.gather(*all_tasks)

        # Stop background metrics collection
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

        # Save vLLM metrics to file
        if vllm_metrics_log:
            try:
                metrics_dir = Path(self.generator_cfg.miniswe_traj_dir) / f"step_{batch_metadata.global_step}"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / "vllm_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(vllm_metrics_log, f, indent=2)
            except Exception:
                pass

        # All outputs now contain valid data (dummy for failed trajectories)
        responses = [output[0] for output in all_outputs]
        rewards = [output[1] for output in all_outputs]
        stop_reasons = [output[2] for output in all_outputs]
        loss_masks = [output[3] for output in all_outputs]
        prompt_token_ids = [output[4] for output in all_outputs]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
