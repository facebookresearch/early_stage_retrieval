"""Train the naive-CF policy via regression."""

from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import hydra

import torch

from experiments.synthetic.function import (
    collect_logged_dataset,
    initialize_trainable_policy,
    initialize_uniform_policy,
    save_logs,
    setup_data_generation_process,
    train_early_stage_with_cf,
)
from experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
    reset_seed,
)
from omegaconf import DictConfig


def _process(
    setting: str,
    key_param: Optional[Union[int, str]],
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    data_size: int,
    dim_model_emb: int,
    early_stage_naive_cf_lr: float,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_eval: int,
    rootdir: str,
    experiment_name: str,
    device: torch.device,
    base_random_seed: int,
    random_seed: int,
    **kwargs,
):
    reset_seed(base_random_seed)

    env = setup_data_generation_process(
        n_user=n_user,
        n_action=n_action,
        n_latent=n_latent,
        n_output_action=n_output_action,
        dim_context=dim_context,
        dim_action_emb=dim_action_emb,
        reward_scaler=reward_scaler,
        device=device,
        random_seed=base_random_seed,
    )

    logging_early_stage_policy, logging_late_stage_policy = initialize_uniform_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )

    reset_seed(random_seed)

    logged_dataset = collect_logged_dataset(
        env=env,
        logging_early_stage_policy=logging_early_stage_policy,
        logging_late_stage_policy=logging_late_stage_policy,
        is_deterministic_early_stage=False,
        is_deterministic_late_stage=False,
        n_candidate_action=100,
        data_size=data_size,
    )

    naive_cf_early_stage_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    naive_cf_early_stage_policy, _, naive_cf_training_logs = train_early_stage_with_cf(
        env=env,
        early_stage_policy=naive_cf_early_stage_policy,
        logged_dataset=logged_dataset,
        early_stage_lr=early_stage_naive_cf_lr,
        model_selector_lr=0.0,  # unused
        n_epoch=n_epoch,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action_eval=n_candidate_action_eval,
        device=device,
        random_seed=random_seed,
    )
    save_logs(
        rootdir=rootdir,
        experiment_name=experiment_name,
        logging_type="uniform",
        credit_assignment_type=None,
        n_candidate_action_train=None,
        setting=setting,
        key_param=key_param,
        random_seed=random_seed,
        trained_naive_cf_early_stage_policy=naive_cf_early_stage_policy,
        naive_cf_training_logs=naive_cf_training_logs,
    )


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param = conf["setting"]
    experiment_name = conf["experiment_name"]

    rootdir = conf["rootdir"]

    if experiment_name == "auto":
        experiment_name = setting

    rootdir = f"{rootdir}/{experiment_name}"

    if conf["early_stage_logging_path"] == "auto":
        conf_["early_stage_logging_path"] = f"{rootdir}/logging/early_stage_policy.pt"

    if conf["late_stage_logging_path"] == "auto":
        conf_["late_stage_logging_path"] = f"{rootdir}/logging/late_stage_policy.pt"

    if not Path(conf_["early_stage_logging_path"]).exists():
        raise ValueError("early_stage_logging_path does not exist.")

    if not Path(conf_["late_stage_logging_path"]).exists():
        raise ValueError("late_stage_logging_path does not exist.")

    if setting != "default":
        key_param_name = conf["setting"]

        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            for random_seed in range(conf["n_random_seed"]):
                conf_["random_seed"] = random_seed + conf["start_random_seed"]

                print(
                    f"Setting: {setting}, Key Param: {key_param}, Random Seed: {random_seed}/{conf['n_random_seed']}"
                )
                _process(**conf_)

    else:
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            print(
                f"Setting: {setting}, Key Param: {'None'}, Random Seed: {random_seed}/{conf['n_random_seed']}"
            )
            _process(**conf_)


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
        "n_candidate_action_train": cfg.setting.n_candidate_action_train,
        "n_candidate_action_eval": cfg.setting.n_candidate_action_eval,
        "n_user": cfg.setting.n_user,
        "n_latent": cfg.setting.n_latent,
        "dim_context": cfg.setting.dim_context,
        "dim_action_emb": cfg.setting.dim_action_emb,
        "reward_scaler": cfg.setting.reward_scaler,
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
