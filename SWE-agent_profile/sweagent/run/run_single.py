"""[cyan][bold]Run SWE-agent on a single instance taken from github or similar.[/bold][/cyan]

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit
  --config CONFIG     Load additional config files. Use this option multiple times to load
                      multiple files, e.g., --config config1.yaml --config config2.yaml

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]github issue[/bold][/cyan][green]:

sweagent run --config config/default.yaml --agent.model.name "gpt-4o" \\
    --env.repo.github_url=https://github.com/SWE-agent/test-repo/ \\
    --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1
[/green]

By default this will start a docker container and run the agent in there.
You can set the image with [green]--env.docker.image[/green].

Here's an example that uses [bold][cyan]modal[/bold][/cyan] instead of docker and also a [bold][cyan]local repository[/bold][/cyan]:

[green]sweagent run --config config/default.yaml --agent.model.name "gpt-4o" \\
    --env.deployment.type=modal --env.repo.path /path/to/repo \\
    --problem_statement.path=path/to/problem_statement.md
[/green]
"""

import getpass
import sys
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sweagent.agent.agents import AbstractAgent, AgentConfig, get_agent_from_config
from sweagent.agent.problem_statement import (
    EmptyProblemStatement,
    ProblemStatement,
    ProblemStatementConfig,
)
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.run.common import AutoCorrectSuggestion as ACS
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.hooks.open_pr import OpenPRConfig, OpenPRHook
from sweagent.utils.config import load_environment_variables
from sweagent.utils.gpu import GPUMonitor
from sweagent.utils.log import add_file_handler, get_logger
from sweagent.utils.profiling import ProfilingManager, maybe_time_stage


class RunSingleActionConfig(BaseModel):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""

    # Open a PR with the patch if we can solve the issue
    open_pr: bool = False
    pr_config: OpenPRConfig = Field(default_factory=OpenPRConfig)
    # When working with local repository: Apply patch
    apply_patch_locally: bool = False

    # pydantic config
    model_config = ConfigDict(extra="forbid")


def _get_default_output_dir(output_dir: Path, problem_statement: ProblemStatement, agent: AgentConfig) -> Path:
    if output_dir == Path("DEFAULT"):
        user_id = getpass.getuser()
        problem_id = problem_statement.id
        try:
            model_id = agent.model.id  # type: ignore[attr-defined]
        except AttributeError:
            model_id = "unknown_model"
        config_file = getattr(agent, "_config_files", ["no_config"])[0]
        if isinstance(config_file, Path):
            config_file = config_file.stem
        return Path.cwd() / "trajectories" / user_id / f"{config_file}__{model_id}___{problem_id}"
    return output_dir


class RunSingleConfig(BaseSettings, cli_implicit_flags=False):
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment options.")
    agent: AgentConfig = Field(description="Agent options.")
    problem_statement: ProblemStatementConfig = Field(
        default_factory=EmptyProblemStatement, description="Problem statement options."
    )
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")

    actions: RunSingleActionConfig = Field(default_factory=RunSingleActionConfig)

    env_var_path: Path | None = None
    """Path to a .env file to load environment variables from."""

    # pydantic config
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")

    def set_default_output_dir(self) -> None:
        # Needs to be called explicitly, because self._config_files will be setup
        # post-init.
        self.output_dir = _get_default_output_dir(self.output_dir, self.problem_statement, self.agent)

    @classmethod
    def _get_auto_correct(cls) -> list[ACS]:
        return [
            ACS("model", "agent.model.name"),
            ACS("agent.model", "agent.model.name"),
            ACS("model.name", "agent.model.name"),
            ACS("per_instance_cost_limit", "agent.model.per_instance_cost_limit"),
            ACS("model.per_instance_cost_limit", "agent.model.per_instance_cost_limit"),
            ACS("config_file", "config"),
            ACS(
                "data_path",
                help="--data_path is no longer support for SWE-A 1.0. Please check the tutorial and use one of the --problem_statement options, e.g., --problem_statement.github_url or --problem_statement.path",
            ),
            ACS(
                "repo_path",
                help="--repo_path is no longer support for SWE-A 1.0. Please check the tutorial and use one of the --env.repo options, e.g., --env.repo.github_url or --env.repo.path",
            ),
            ACS("repo.path", "env.repo.path"),
        ]


class RunSingle:
    def __init__(
        self,
        env: SWEEnv,
        agent: AbstractAgent,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        *,
        output_dir: Path = Path("."),
        hooks: list[RunHook] | None = None,
        actions: RunSingleActionConfig | None = None,
        profiler: ProfilingManager | None = None,
    ):
        """Note: When initializing this class, make sure to add the hooks that are required by your actions.
        See `from_config` for an example.
        """
        self.logger = get_logger("swea-run", emoji="🏃")
        instance_id = problem_statement.id
        _log_filename_template = f"{instance_id}.{{level}}.log"
        for level in ["trace", "debug", "info"]:
            add_file_handler(
                output_dir / instance_id / _log_filename_template.format(level=level),
                level=level,
                id_=f"{instance_id}-{level}",
            )
        self.env = env
        self.agent = agent
        self.output_dir = output_dir
        self._hooks = []
        self.profiler = profiler
        if actions is not None:
            actions = RunSingleActionConfig()
        self.actions = actions
        self._chooks = CombinedRunHooks()
        self.problem_statement = problem_statement
        for hook in hooks or []:
            self.add_hook(hook)

    @property
    def hooks(self) -> list[RunHook]:
        return self._chooks.hooks

    @classmethod
    def from_config(cls, config: RunSingleConfig) -> Self:
        profiler = ProfilingManager(problem_id=config.problem_statement.id)
        with profiler.time_stage("config_load"):
            load_environment_variables(config.env_var_path)
            config.set_default_output_dir()
            config.output_dir.mkdir(parents=True, exist_ok=True)
            agent = get_agent_from_config(config.agent)
            agent.replay_config = config  # type: ignore[attr-defined]
        profiler.set_output_path(config.output_dir / config.problem_statement.id / "profiling.json")
        profiler.register_group(
            "observation_packaging",
            ("observation_get_state", "observation_postprocess"),
        )
        profiler.register_group(
            "agent_run",
            (
                "agent_setup",
                "lm_reasoning",
                "tool_parse_validate",
                "env_execution",
                "observation_packaging",
                "retry_handling",
                "finalization",
            ),
        )
        self = cls(
            env=SWEEnv.from_config(config.env),
            agent=agent,
            problem_statement=config.problem_statement,
            output_dir=config.output_dir,
            actions=config.actions,
            profiler=profiler,
        )
        self.add_hook(SaveApplyPatchHook(apply_patch_locally=config.actions.apply_patch_locally))
        if config.actions.open_pr:
            self.logger.debug("Adding OpenPRHook")
            self.add_hook(OpenPRHook(config.actions.pr_config))
        return self

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def run(self):
        self._chooks.on_start()
        output_dir = self.output_dir / self.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.profiler is not None:
            self.profiler.set_output_path(output_dir / "profiling.json")
            self.profiler.add_metadata("mode", "run_single")
        result = None
        env_started = False
        gpu_monitor = GPUMonitor()
        gpu_monitor_started = gpu_monitor.start()
        try:
            self.logger.info("Starting environment")
            with maybe_time_stage(self.profiler, "env_preparation"):
                self.env.start()
            env_started = True
            self.logger.info("Running agent")
            self._chooks.on_instance_start(index=0, env=self.env, problem_statement=self.problem_statement)
            with maybe_time_stage(self.profiler, "other"):
                if self.agent.replay_config is not None:  # type: ignore[attr-defined]
                    (output_dir / "config.yaml").write_text(
                        yaml.dump(self.agent.replay_config.model_dump_json(), indent=2)
                    )  # type: ignore[attr-defined]
                if hasattr(self.agent, "profiler"):
                    self.agent.profiler = self.profiler  # type: ignore[attr-defined]
                if hasattr(self.agent, "gpu_monitor"):
                    self.agent.gpu_monitor = gpu_monitor  # type: ignore[attr-defined]
            with maybe_time_stage(self.profiler, "agent_run"):
                result = self.agent.run(
                    problem_statement=self.problem_statement,
                    env=self.env,
                    output_dir=output_dir,
                )
        except Exception:
            if self.profiler is not None:
                self.profiler.set_status("failure")
            raise
        else:
            self._chooks.on_instance_completed(result=result)
            self.logger.info("Done")
            if self.profiler is not None:
                exit_status = result.info.get("exit_status")
                self.profiler.set_exit_status(exit_status)
                success = isinstance(exit_status, str) and exit_status.startswith("submitted")
                self.profiler.set_status("success" if success else "failure")
                stats = result.info.get("model_stats", {})
                if isinstance(stats, dict):
                    for key in ("tokens_sent", "tokens_received", "api_calls", "instance_cost"):
                        if key in stats:
                            self.profiler.add_metadata(key, stats[key])
                stage_stats = result.info.get("stage_stats")
                if isinstance(stage_stats, dict):
                    self.profiler.add_metadata("stage_stats", stage_stats)
                    gpu_segments = {
                        stage: stats["gpu_metrics"]
                        for stage, stats in stage_stats.items()
                        if isinstance(stats, dict) and stats.get("gpu_metrics")
                    }
                    if gpu_segments:
                        self.profiler.add_metadata("gpu_metrics_segments", gpu_segments)
                    prefill_segments: dict[str, int] = {}
                    decode_segments: dict[str, int] = {}
                    step_segments: dict[str, int] = {}
                    for stage, stats in stage_stats.items():
                        if not isinstance(stats, dict):
                            continue
                        if "prefill_tokens" in stats and isinstance(stats["prefill_tokens"], (int, float)):
                            prefill_segments[stage] = int(stats["prefill_tokens"])
                        if "decode_tokens" in stats and isinstance(stats["decode_tokens"], (int, float)):
                            decode_segments[stage] = int(stats["decode_tokens"])
                        if "steps" in stats and isinstance(stats["steps"], (int, float)):
                            step_segments[stage] = int(stats["steps"])
                    if prefill_segments:
                        self.profiler.add_metadata("prefill_tokens_segments", prefill_segments)
                    if decode_segments:
                        self.profiler.add_metadata("decode_tokens_segments", decode_segments)
                    if step_segments:
                        self.profiler.add_metadata("step_count_segments", step_segments)
            save_predictions(self.output_dir, self.problem_statement.id, result)
        finally:
            self._chooks.on_end()
            if env_started:
                with maybe_time_stage(self.profiler, "env_shutdown"):
                    self.env.close()
            gpu_metrics: dict[str, object] | None = None
            if gpu_monitor_started:
                gpu_metrics = gpu_monitor.stop()
            if self.profiler is not None:
                if gpu_metrics:
                    self.profiler.add_metadata("gpu_metrics", gpu_metrics)
                elif not gpu_monitor_started:
                    self.profiler.add_metadata("gpu_metrics", {"available": False})
            if self.profiler is not None:
                self.profiler.dump()
        return result


def run_from_config(config: RunSingleConfig):
    RunSingle.from_config(config).run()


def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (  # type: ignore
        __doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunSingleConfig)
    )
    run_from_config(BasicCLI(RunSingleConfig, help_text=help_text).get_config(args))  # type: ignore


if __name__ == "__main__":
    run_from_cli()

