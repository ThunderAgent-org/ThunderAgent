"""
Main entrypoint for fully async mini-SWE-agent training.

Combines MiniSweAgentGenerator (multi-turn Docker-based SWE agent) with
FullyAsyncRayPPOTrainer (overlapping generation and training).
"""

import asyncio

import hydra
import ray
from omegaconf import DictConfig, OmegaConf, open_dict

from skyrl_train.entrypoints.main_base import config_dir, validate_cfg
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.utils import initialize_ray

from .main_mini_swe import MiniSWEPPOExp


class MiniSWEFullyAsyncPPOExp(MiniSWEPPOExp):
    """Mini-SWE-agent experiment with fully async training."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    import os
    import sys
    import traceback

    model_path = cfg.trainer.policy.model.path

    # If model_path is already a local directory (e.g., flat copy at /scratch/models/...),
    # use it directly. Otherwise, resolve via snapshot_download.
    if os.path.isdir(model_path):
        local_path = model_path
    else:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(model_path)

    with open_dict(cfg):
        cfg.trainer.policy.model.path = local_path

    try:
        exp = MiniSWEFullyAsyncPPOExp(cfg)
        exp.run()
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"FATAL: skyrl_entrypoint crashed: {e}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        raise


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    import sys
    import traceback

    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"FATAL: Training failed: {e}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
