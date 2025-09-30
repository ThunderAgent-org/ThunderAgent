from pathlib import Path
from typing import cast

import pytest

from sweagent.run.run import main
from sweagent.run.run_batch import RunBatch
from sweagent.run.batch_instances import BatchInstance
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.utils.log import get_logger
from swerex.deployment.config import DockerDeploymentConfig


@pytest.mark.slow
def test_expert_instances(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "expert_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.type",
        "expert_file",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--raise_exceptions",
        "True",
    ]
    main(cmd)
    for _id in ["simple_test_problem", "simple_test_problem_2"]:
        assert (tmp_path / f"{_id}" / f"{_id}.traj").exists(), list(tmp_path.iterdir())


@pytest.mark.slow
def test_simple_instances(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "simple_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--raise_exceptions",
        "True",
    ]
    main(cmd)
    assert (tmp_path / "simple_test_problem" / "simple_test_problem.traj").exists(), list(tmp_path.iterdir())


def test_empty_instances_simple(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "simple_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--raise_exceptions",
        "True",
        "--instances.filter",
        "doesnotmatch",
    ]
    with pytest.raises(ValueError, match="No instances to run"):
        main(cmd)


def test_cleanup_environment_resources_invokes_docker(monkeypatch):
    rb = RunBatch.__new__(RunBatch)
    rb.logger = get_logger("test-run-batch-cleanup")

    recorded: list[tuple[tuple[str, ...], str]] = []

    monkeypatch.setattr(rb, "_list_containers_for_image", lambda image: ["container-a", "container-b"])

    def _record(args, description):
        recorded.append((tuple(args), description))

    monkeypatch.setattr(rb, "_run_docker_command", _record)

    deployment_cfg = DockerDeploymentConfig(image="example:test")
    env_cfg = EnvironmentConfig(deployment=deployment_cfg)
    problem = TextProblemStatement(text="example", id="example__id")
    instance = BatchInstance(env=env_cfg, problem_statement=problem)

    rb._cleanup_environment_resources(instance, cast(object, None))

    assert recorded == [
        (("docker", "rm", "-f", "container-a"), "remove container container-a for example__id"),
        (("docker", "rm", "-f", "container-b"), "remove container container-b for example__id"),
        (("docker", "image", "rm", "-f", "example:test"), "remove image example:test for example__id"),
    ]


def test_empty_instances_expert(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "expert_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--instances.type",
        "expert_file",
        "--output_dir",
        str(tmp_path),
        "--raise_exceptions",
        "True",
        "--instances.filter",
        "doesnotmatch",
    ]
    with pytest.raises(ValueError, match="No instances to run"):
        main(cmd)


# This doesn't work because we need to retrieve environment variables from the environment
# in order to format our templates.
# def test_run_batch_swe_bench_instances(tmp_path: Path):
#     cmd = [
#         "run-batch",
#         "--agent.model.name",
#         "instant_empty_submit",
#         "--instances.subset",
#         "lite",
#         "--instances.split",
#         "test",
#         "--instances.slice",
#         "0:1",
#         "--output_dir",
#         str(tmp_path),
#         "--raise_exceptions",
#         "--instances.deployment.type",
#         "dummy",
#     ]
#     main(cmd)
