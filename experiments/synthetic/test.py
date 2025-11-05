"""File to test .py file execution on kernel and manifold."""

from pathlib import Path
from time import time
from typing import Any, Dict

import hydra

import torch

from experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
)

from synthetic.dataset import (
    SyntheticDataGenerator,
)
from omegaconf import DictConfig

# relative import is somehow not working
# from .utils import assert_configuration, format_runtime


def process(
    conf: Dict[str, Any],
):
    print("Hello world!")

    tmp_tensor = torch.tensor([1, 2, 3])
    print(tmp_tensor)

    env = SyntheticDataGenerator()
    print("env initialized successfully..")

    rootdir = conf["rootdir"]
    rootdir = Path(f"{rootdir}/test")

    rootdir.mkdir(parents=True, exist_ok=True)

    original_local_path = Path(rootdir / "original_tensor.pt")
    torch.save(tmp_tensor, original_local_path)


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    assert_configuration(cfg)
    print(f"The current working directory is: {Path().cwd()}")
    print(f"The original working directory is: {hydra.utils.get_original_cwd()}")
    print()

    conf = {
        "setting": cfg.setting.setting,
        "data_size": cfg.setting.data_size,
        "n_action": cfg.setting.n_action,
        "n_output_action": cfg.setting.n_output_action,
        "n_candidate_action_eval": cfg.setting.n_candidate_action_eval,
        "n_user": cfg.setting.n_user,
        "n_latent": cfg.setting.n_latent,
        "dim_context": cfg.setting.dim_context,
        "dim_action_emb": cfg.setting.dim_action_emb,
        "reward_scaler": cfg.setting.reward_scaler,
        "logging_type": cfg.setting.logging_type,
        "device": cfg.setting.device,
        "n_random_seed": cfg.setting.n_random_seed,
        "start_random_seed": cfg.setting.start_random_seed,
        "base_random_seed": cfg.setting.base_random_seed,
        "dim_model_emb": cfg.model.dim_model_emb,
        "n_moe_model": cfg.model.n_moe_model,
        "early_stage_naive_cf_lr": cfg.model.early_stage_naive_cf_lr,
        "late_stage_neural_lr": cfg.model.late_stage_neural_lr,
        "online_vanilla_pg_lr": cfg.model.online_vanilla_pg_lr,
        "online_credit_assigned_pg_lr": cfg.model.online_credit_assigned_pg_lr,
        "credit_assignment_type": cfg.model.credit_assignment_type,
        "n_epoch": cfg.model.n_epoch,
        "n_steps_per_epoch": cfg.model.n_steps_per_epoch,
        "n_epochs_per_log": cfg.model.n_epochs_per_log,
        "rootdir": cfg.logs.rootdir,
        "experiment_name": cfg.logs.experiment_name,
        "early_stage_naive_cf_path": cfg.path.early_stage_naive_cf_path,  # unused
        "late_stage_naive_cf_path": cfg.path.late_stage_naive_cf_path,  # unused
        "early_stage_online_credit_assigned_pg_path": cfg.path.early_stage_online_credit_assigned_pg_path,
        "early_stage_online_vanilla_pg_path": cfg.path.early_stage_online_vanilla_pg_path,
    }
    process(conf)


if __name__ == "__main__":
    start = time()
    main()
    finish = time()
    print(f"Total runtime: {format_runtime(start, finish)}")
