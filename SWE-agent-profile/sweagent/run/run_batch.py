"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\ # configure instances
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\     # first 50 instances
    --instances.shuffle=True \\  # shuffle instances (with fixed seed)
    --config config/default.yaml \\
    --agent.model.name gpt-4o  # configure model
[/green]

[cyan][bold]=== LOADING INSTANCES ===[/bold][/cyan]

[cyan][bold]From a file[/bold][/cyan] [green]--instances.type file --instances.path /path/to/file[/green].
[cyan][bold]From huggingface[/bold][/cyan] [green]--instances.type huggingface --instances.dataset_name=SWE_Bench_lite --instances.split=dev[/green].

All instance specifications support the [green]filter[/green], [green]slice[/green], and [green]shuffle[/green] options.
With [green]filter[/green], you can select specific instances, e.g., [green]--instances.filter='instance_id_1|instance_id_2'[/green].
"""

import getpass
import json
import logging
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import Self

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.live import Live
from swerex.deployment.hooks.status import SetStatusDeploymentHook

from sweagent import TRAJECTORY_DIR
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.agent.hooks.stage_timing import StageTimingAgentHook
from sweagent.agent.hooks.prefix_cache_metrics import PrefixCacheMetricsHook
from sweagent.agent.hooks.gpu_monitor import GpuMetricsHook
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.agent.hooks.step_sync import StepSyncAgentHook
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.run_single import RunSingleConfig
from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.stage_timing import StageTimingLogger
from sweagent.utils.step_sync import StepSynchronizer
from sweagent.utils.log import (
    add_file_handler,
    add_logger_names_to_stream_handlers,
    get_logger,
    register_thread_name,
    remove_file_handler,
    set_stream_handler_levels,
)


class RunBatchConfig(BaseSettings, cli_implicit_flags=False):
    instances: BatchInstanceSourceConfig = Field(description="Instances to run.")
    agent: AgentConfig = Field(description="Agent options.")
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")
    suffix: str = ""
    """Suffix to add to the output directory. Only used if `output_dir` is `DEFAULT`."""
    raise_exceptions: bool = False
    """Raise exceptions instead of skipping instances."""
    redo_existing: bool = False
    """Do not skip instances that already have a trajectory."""
    env_var_path: Path | None = None
    """Path to a .env file to load environment variables from."""
    num_workers: int = Field(default=1)
    """Number of parallel workers to use."""
    random_delay_multiplier: float = 0.3
    """We will wait for a random amount of time between 0 and `random_delay_multiplier`
    times the number of workers at the start of each instance. This is to avoid any
    potential race condition or issues with bottlenecks, e.g., when running on a platform
    with few CPUs that cannot handle the startup of all containers in time.
    """
    progress_bar: bool = True
    """Whether to show a progress bar. Progress bar is never shown for human models.
    Progress bar is always shown for multi-worker runs.
    """

    # pydantic config
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")

    def set_default_output_dir(self) -> None:
        # Needs to be called explicitly, because self._config_files will be setup
        # post-init.
        if self.output_dir == Path("DEFAULT"):
            user_id = getpass.getuser()
            source_id = self.instances.id
            try:
                model_id = self.agent.model.id  # type: ignore[attr-defined]
            except AttributeError:
                model_id = "unknown"
            config_file = getattr(self, "_config_files", ["no_config"])[0]
            if config_file != "no_config":
                config_file = Path(config_file).stem
            suffix = f"__{self.suffix}" if self.suffix else ""
            self.output_dir = TRAJECTORY_DIR / user_id / f"{config_file}__{model_id}___{source_id}{suffix}"

    @model_validator(mode="after")
    def evaluate_and_redo_existing(self) -> Self:
        if not isinstance(self.instances, SWEBenchInstances):
            return self
        if self.instances.evaluate and self.redo_existing:
            msg = (
                "Cannot evaluate and redo existing at the same time. This would cause invalid results, because "
                "after the first merge_preds gives you a preds.json, this file would be submitted to SB-CLI, causing"
                "evaluation of old instances, which could then not be overwritten by the new ones."
            )
            raise ValueError(msg)
        return self


class _BreakLoop(Exception):
    """Used for internal control flow"""


class RunBatch:
    def __init__(
        self,
        instances: list[BatchInstance],
        agent_config: AgentConfig,
        *,
        output_dir: Path = Path("."),
        hooks: list[RunHook] | None = None,
        raise_exceptions: bool = False,
        redo_existing: bool = False,
        num_workers: int = 1,
        progress_bar: bool = True,
        random_delay_multiplier: float = 0.3,
    ):
        """Note: When initializing this class, make sure to add the hooks that are required by your actions.
        See `from_config` for an example.

        Args:
            hooks: If not specified, the default hooks will be used.
            num_workers: Number of parallel workers to use. Default is 1 (sequential execution).
            progress_bar: Whether to show a progress bar. Progress bar is never shown for human models.
                Progress bar is always shown for multi-worker runs.
            random_delay_multiplier: We will wait for a random amount of time between 0 and `random_delay_multiplier`
                times the number of workers at the start of each instance. This is to avoid any
                potential race conditions.
        """
        if self._model_id in ["human", "human_thought"] and num_workers > 1:
            msg = "Cannot run with human model in parallel"
            raise ValueError(msg)

        self.logger = get_logger("swea-run", emoji="🏃")
        add_file_handler(
            output_dir / "run_batch.log",
            id_="progress",
            filter=lambda name: "swea-run" in name or "config" in name,
        )
        self.instances = instances
        self.agent_config = agent_config
        self.output_dir = output_dir
        self._raise_exceptions = raise_exceptions
        self._chooks = CombinedRunHooks()
        self._redo_existing = redo_existing
        self._num_workers = min(num_workers, len(instances))
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(
            num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml"
        )
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier
        self._resource_lock = Lock()
        self._instance_resources: dict[str, dict[str, set[str]]] = {}

    @property
    def _model_id(self) -> str:
        try:
            return self.agent_config.model.id  # type: ignore[attr-defined]
        except AttributeError:
            return "unknown"

    @classmethod
    def from_config(cls, config: RunBatchConfig) -> Self:
        load_environment_variables(config.env_var_path)
        config.set_default_output_dir()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "run_batch.config.yaml").write_text(yaml.dump(config.model_dump_json(), indent=2))
        logger = get_logger("run", emoji="🏃")
        logger.debug("Loading instances from %s", f"{config.instances!r}")
        instances = config.instances.get_instance_configs()
        logger.info("Loaded %d instances", len(instances))
        if not instances:
            msg = (
                "No instances to run. Here are a few things to check:\n"
                "- With huggingface data: Check that you have the right split (test or dev)\n"
                "- Check your filter does not exclude all instances (check the info log messages)"
            )
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(
            instances=instances,
            agent_config=config.agent,
            output_dir=config.output_dir,
            raise_exceptions=config.raise_exceptions,
            redo_existing=config.redo_existing,
            num_workers=config.num_workers,
            progress_bar=config.progress_bar,
            random_delay_multiplier=config.random_delay_multiplier,
        )
        if isinstance(config.instances, SWEBenchInstances) and config.instances.evaluate:
            from sweagent.run.hooks.swe_bench_evaluate import SweBenchEvaluate

            rb.add_hook(
                SweBenchEvaluate(
                    output_dir=config.output_dir,
                    subset=config.instances.subset,
                    split=config.instances.split,
                    continuous_submission_every=30,
                )
            )
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()

        if self._num_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()

        output_dirs = []
        for instance in self.instances:
            output_dirs.append(self.output_dir / instance.problem_statement.id)
        merge_predictions(output_dirs, self.output_dir / "preds.json")

        self._chooks.on_end()

    def main_single_worker(self) -> None:
        with ExitStack() as stack:
            # Conditionally add progress bar
            if self._model_id not in ["human", "human_thought"] and self._show_progress_bar:
                stack.enter_context(Live(self._progress_manager.render_group))
            for instance in self.instances:
                try:
                    self.run_instance(instance)
                except _BreakLoop:
                    self.logger.info("Stopping loop over instances")
                    break

    def main_multi_worker(self) -> None:
        add_logger_names_to_stream_handlers()
        # Set all stream handlers to WARNING and set everything where we want to have
        # more verbosity explicitly
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)  # type: ignore

        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                stop = False
                try:
                    for start in range(0, len(self.instances), self._num_workers):
                        if stop:
                            break
                        batch = self.instances[start : start + self._num_workers]
                        synchronizer = StepSynchronizer(
                            instance.problem_statement.id for instance in batch
                        )
                        futures = [
                            executor.submit(self.run_instance, instance, synchronizer)
                            for instance in batch
                        ]
                        try:
                            for future in as_completed(futures):
                                future.result()
                        except (KeyboardInterrupt, _BreakLoop):
                            msg = (
                                "Received keyboard interrupt, waiting for running instances "
                                "to finish, but cancelled everything else"
                            )
                            self.logger.info(msg)
                            for future in futures:
                                future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            stop = True
                            break
                        finally:
                            self._cleanup_batch_resources(batch)
                        if stop:
                            break
                        if start + len(batch) < len(self.instances):
                            self.logger.info(
                                "Batch completed, throttling for 60 seconds before next batch after cleanup"
                            )
                            time.sleep(60)
                finally:
                    self._progress_manager.print_report()

    def run_instance(self, instance: BatchInstance, synchronizer: StepSynchronizer | None = None) -> None:
        self.logger.info("Running on instance %s", instance.problem_statement.id)
        register_thread_name(instance.problem_statement.id)
        self._add_instance_log_file_handlers(instance.problem_statement.id, multi_worker=self._num_workers > 1)
        # Let's add some randomness to avoid any potential race conditions or thundering herd
        if self._progress_manager.n_completed < self._num_workers:
            time.sleep(random.random() * self._random_delay_multiplier * (self._num_workers - 1))

        self._progress_manager.on_instance_start(instance.problem_statement.id)

        if previous_exit_status := self.should_skip(instance):
            self._progress_manager.on_instance_end(
                instance.problem_statement.id, exit_status=f"skipped ({previous_exit_status})"
            )
            if synchronizer is not None:
                synchronizer.mark_done(instance.problem_statement.id)
            self._remove_instance_log_file_handlers(instance.problem_statement.id)
            return

        # Either catch and silence exception, or raise _BreakLoop to stop the loop
        # over the instances
        try:
            result = self._run_instance(instance, synchronizer=synchronizer)
        except KeyboardInterrupt:
            raise _BreakLoop
        except (SystemExit, ModelConfigurationError, TotalCostLimitExceededError) as e:
            if self._raise_exceptions:
                raise
            self.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"❌ Failed on {instance.problem_statement.id}: {e}")
            self._progress_manager.on_uncaught_exception(instance.problem_statement.id, e)
            if self._raise_exceptions:
                raise
        else:
            self._progress_manager.on_instance_end(
                instance.problem_statement.id, exit_status=result.info.get("exit_status", "unknown_exit")
            )
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance.problem_statement.id)

    def _run_instance(
        self,
        instance: BatchInstance,
        *,
        synchronizer: StepSynchronizer | None = None,
    ) -> AgentRunResult:
        output_dir = Path(self.output_dir) / instance.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_config.name = f"{instance.problem_statement.id}"
        agent = get_agent_from_config(self.agent_config)
        single_run_replay_config = RunSingleConfig(
            agent=self.agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
        (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
        )
        agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
        image_name = getattr(instance.env.deployment, "image", None)
        if isinstance(image_name, str) and image_name:
            self._register_instance_resources(instance.problem_statement.id, images={image_name})
        timing_logger = StageTimingLogger(
            output_dir / "stage_timings.jsonl",
            instance_id=instance.problem_statement.id,
        )
        stage_timing_hook = StageTimingAgentHook(timing_logger)
        agent.add_hook(stage_timing_hook)
        agent.add_hook(PrefixCacheMetricsHook())
        agent.add_hook(GpuMetricsHook())
        if synchronizer is not None:
            agent.add_hook(StepSyncAgentHook(instance.problem_statement.id, synchronizer))
        agent.add_hook(SetStatusAgentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        self._progress_manager.update_instance_status(instance.problem_statement.id, "Starting environment")
        instance.env.name = f"{instance.problem_statement.id}"
        env = SWEEnv.from_config(instance.env)
        env.add_hook(
            SetStatusEnvironmentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        env.deployment.add_hook(
            SetStatusDeploymentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        stage_timing_hook.start_env_prepare()
        try:
            env.start()
            container_identifiers = self._extract_container_identifiers(env)
            if container_identifiers:
                self._register_instance_resources(
                    instance.problem_statement.id,
                    containers=container_identifiers,
                )
            self._chooks.on_instance_start(index=0, env=env, problem_statement=instance.problem_statement)
            stage_timing_hook.finish_env_prepare()
            result = agent.run(
                problem_statement=instance.problem_statement,
                env=env,
                output_dir=output_dir,
            )
        except Exception:
            stage_timing_hook.finish_env_prepare(extra={"reason": "exception"})
            # The actual handling is happening in `run_instance`, but we need to make sure that
            # we log it to the agent specific logger as well
            agent.logger.error(traceback.format_exc())  # type: ignore[attr-defined]
            raise
        finally:
            env.close()
            if synchronizer is not None:
                synchronizer.mark_done(instance.problem_statement.id)
        save_predictions(self.output_dir, instance.problem_statement.id, result)
        self._chooks.on_instance_completed(result=result)
        return result

    def should_skip(self, instance: BatchInstance) -> bool | str:
        """Check if we should skip this instance.
        Returns previous exit status if the instance should be skipped.
        """
        if self._redo_existing:
            return False

        # Check if there's an existing trajectory for this instance
        log_path = self.output_dir / instance.problem_statement.id / (instance.problem_statement.id + ".traj")
        if not log_path.exists():
            return False

        content = log_path.read_text()
        if not content.strip():
            self.logger.warning("Found empty trajectory: %s. Removing.", log_path)
            log_path.unlink()
            return False

        try:
            data = json.loads(content)
            # If the trajectory has no exit status, it's incomplete and we will redo it
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Found existing trajectory with no exit status: {log_path}. Removing.")
                log_path.unlink()
                return False
        except Exception as e:
            self.logger.error(f"Failed to check existing trajectory: {log_path}: {e}. Removing.")
            # If we can't check the trajectory, we will redo it
            log_path.unlink()
            return False
        # otherwise, we will skip it
        self.logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
        return exit_status

    def _add_instance_log_file_handlers(self, instance_id: str, multi_worker: bool = False) -> None:
        filename_template = f"{instance_id}.{{level}}.log"
        for level in ["trace", "debug", "info"]:
            filter = instance_id if multi_worker else ""
            add_file_handler(
                self.output_dir / instance_id / filename_template.format(level=level),
                filter=filter,
                level=level,
                id_=f"{instance_id}-{level}",
            )

    def _remove_instance_log_file_handlers(self, instance_id: str) -> None:
        for level in ["trace", "debug", "info"]:
            remove_file_handler(f"{instance_id}-{level}")

    def _register_instance_resources(
        self,
        instance_id: str,
        *,
        containers: set[str] | None = None,
        images: set[str] | None = None,
    ) -> None:
        if not containers and not images:
            return
        with self._resource_lock:
            resource_entry = self._instance_resources.setdefault(instance_id, {"containers": set(), "images": set()})
            if containers:
                resource_entry["containers"].update(containers)
            if images:
                resource_entry["images"].update(images)

    def _cleanup_batch_resources(self, batch: list[BatchInstance]) -> None:
        if not batch:
            return
        if shutil.which("docker") is None:
            self.logger.debug("Docker executable not found; skipping cleanup for current batch")
            return
        containers: set[str] = set()
        images: set[str] = set()
        with self._resource_lock:
            for instance in batch:
                instance_id = instance.problem_statement.id
                info = self._instance_resources.pop(instance_id, None)
                if info:
                    containers.update(info.get("containers", set()))
                    images.update(info.get("images", set()))
                image_name = getattr(instance.env.deployment, "image", None)
                if isinstance(image_name, str) and image_name:
                    images.add(image_name)
        if not containers and not images:
            self.logger.debug("No docker resources recorded for batch; skipping cleanup")
            return
        self.logger.info(
            "Cleaning docker resources for batch (%d containers, %d images)", len(containers), len(images)
        )
        for image in list(images):
            containers.update(self._list_containers_for_image(image))
        for container in containers:
            self._run_docker_command(["docker", "rm", "-f", container], ignore_errors=True)
        for image in images:
            self._run_docker_command(["docker", "image", "rm", image], ignore_errors=True)
        self._run_docker_command(["docker", "container", "prune", "-f"], ignore_errors=True)
        self._run_docker_command(["docker", "image", "prune", "-f"], ignore_errors=True)

    def _extract_container_identifiers(self, env: SWEEnv) -> set[str]:
        identifiers: set[str] = set()
        deployment = getattr(env, "deployment", None)
        if deployment is not None:
            for attr in ("container_name", "name"):
                value = getattr(deployment, attr, None)
                if isinstance(value, str) and value:
                    identifiers.add(value)
            runtime = getattr(deployment, "runtime", None)
            if runtime is not None:
                for attr in ("container_name", "name"):
                    value = getattr(runtime, attr, None)
                    if isinstance(value, str) and value:
                        identifiers.add(value)
        return identifiers

    def _list_containers_for_image(self, image: str) -> set[str]:
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"ancestor={image}", "--format", "{{.ID}}"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return set()
        if result.returncode != 0:
            if result.stderr.strip():
                self.logger.debug(
                    "Failed to list containers for image %s: %s", image, result.stderr.strip()
                )
            return set()
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _run_docker_command(self, command: list[str], *, ignore_errors: bool = False) -> None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            self.logger.warning("Docker executable not found while running cleanup command; skipping")
            return
        cmd_display = " ".join(command)
        if completed.stdout.strip():
            self.logger.debug("Docker command output [%s]: %s", cmd_display, completed.stdout.strip())
        if completed.returncode != 0:
            if ignore_errors:
                if completed.stderr.strip():
                    self.logger.debug(
                        "Docker command failed (ignored) [%s]: %s", cmd_display, completed.stderr.strip()
                    )
            else:
                self.logger.warning(
                    "Docker command failed [%s] (code %s): %s",
                    cmd_display,
                    completed.returncode,
                    completed.stderr.strip(),
                )

def run_from_config(config: RunBatchConfig):
    RunBatch.from_config(config).main()


def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (  # type: ignore
        __doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunBatchConfig)
    )
    run_from_config(BasicCLI(RunBatchConfig, help_text=help_text).get_config(args))  # type: ignore


if __name__ == "__main__":
    run_from_cli()
