"""Evaluate oracle, uniform, and logging on synthetic simulation."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Union

import hydra
import pandas as pd

import torch

from experiments.synthetic._function import (
    evaluate_policy,
    initialize_optimal_policy,
    initialize_uniform_policy,
    load_logging_policy,
    setup_data_generation_process,
)
from experiments.synthetic._utils import (
    assert_configuration,
    defaultdict_to_dict,
    format_runtime,
    reset_seed,
)
from omegaconf import DictConfig


def _process(
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    dim_model_emb: int,
    n_candidate_action_eval: int,
    device: torch.device,
    base_random_seed: int,
    early_stage_logging_path: str,
    late_stage_logging_path: str,
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

    uniform_early_stage_policy, uniform_late_stage_policy = initialize_uniform_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    optimal_early_stage_policy, optimal_late_stage_policy = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    logging_early_stage_policy, logging_late_stage_policy = load_logging_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_logging_path=early_stage_logging_path,
        late_stage_logging_path=late_stage_logging_path,
        device=device,
        random_seed=base_random_seed,
    )

    algorithms = [
        "logging (uniform)",  # uniform x uniform
        "logging (skewed)",  # skewed x skewed
        "logging (practical)",  # skewed x deficient
        "logging (deficient)",  # deficient x deficient
        "uniform",  # uniform x optimal
        "optimal",  # optimal x optimal
    ]
    early_stage_policies = [
        uniform_early_stage_policy,
        logging_early_stage_policy,
        logging_early_stage_policy,
        logging_early_stage_policy,
        uniform_early_stage_policy,
        optimal_early_stage_policy,
    ]
    late_stage_policies = [
        uniform_late_stage_policy,
        logging_late_stage_policy,
        logging_late_stage_policy,
        logging_late_stage_policy,
        optimal_late_stage_policy,
        optimal_late_stage_policy,
    ]
    is_deterministic_early_stage_flgs = [False, False, False, True, False, True]
    is_derterministic_late_stage_flgs = [False, False, True, True, True, True]

    performance = {}
    for i, algo in enumerate(algorithms):
        performance[algo] = evaluate_policy(
            env=env,
            early_stage_policy=early_stage_policies[i],
            late_stage_policy=late_stage_policies[i],
            n_candidate_action=n_candidate_action_eval,
            is_deterministic_early_stage=is_deterministic_early_stage_flgs[i],
            is_deterministic_late_stage=is_derterministic_late_stage_flgs[i],
        )

    return performance


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
        if setting == "n_candidate_action_eval":
            experiment_name = "default"
        else:
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

    performance_dict = defaultdict(list)

    if setting != "default":
        key_param_name = conf["setting"]

        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            print(f"Setting: {setting}, Key Param: {key_param}")

            performance_ = _process(**conf_)
            performance_dict[key_param_name].append(key_param)

            for key, value in performance_.items():
                performance_dict[key].append(value)

    else:
        print(f"Setting: {setting}, Key Param: {'None'}")

        performance_ = _process(**conf_)
        performance_dict["setting"].append("default")

        for key, value in performance_.items():
            performance_dict[key].append(value)

    performance_dict = defaultdict_to_dict(performance_dict)
    df = pd.DataFrame(performance_dict)

    Path(rootdir).mkdir(parents=True, exist_ok=True)
    df_path = f"{rootdir}/reference_performance.csv"
    df.to_csv(df_path, index=False)


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
        "n_candidate_action_logging": cfg.setting.n_candidate_action_logging,
        "n_candidate_action_train": cfg.setting.n_candidate_action_train,
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
        "early_stage_logging_lr": cfg.model.early_stage_logging_lr,
        "late_stage_logging_lr": cfg.model.late_stage_logging_lr,
        "early_stage_naive_cf_lr": cfg.model.early_stage_naive_cf_lr,
        "early_stage_moe_cf_lr": cfg.model.early_stage_moe_cf_lr,
        "early_stage_moe_selector_lr": cfg.model.early_stage_moe_selector_lr,
        "quantile_cf_lr": cfg.model.quantile_cf_lr,
        "late_stage_neural_lr": cfg.model.late_stage_neural_lr,
        "online_vanilla_pg_lr": cfg.model.online_vanilla_pg_lr,
        "online_credit_assigned_pg_lr": cfg.model.online_credit_assigned_pg_lr,
        "is_vanilla_pg_lr": cfg.model.is_vanilla_pg_lr,
        "is_credit_assigned_pg_lr": cfg.model.is_credit_assigned_pg_lr,
        "kernel_vanilla_pg_lr": cfg.model.kernel_vanilla_pg_lr,
        "kernel_creedit_assigned_pg_lr": cfg.model.kernel_creedit_assigned_pg_lr,
        "logging_action_prob_model_lr": cfg.model.action_prob_model_lr,
        "logging_marginal_model_lr": cfg.model.logging_marginal_model_lr,
        "kernel_bandwidth": cfg.model.kernel_bandwidth,
        "credit_assignment_type": cfg.model.credit_assignment_type,
        "n_epoch": cfg.model.n_epoch,
        "n_epoch_regression": cfg.model.n_epoch_regression,
        "n_epoch_logging": cfg.model.n_epoch_logging,
        "n_steps_per_epoch": cfg.model.n_steps_per_epoch,
        "n_epochs_per_log": cfg.model.n_epochs_per_log,
        "rootdir": cfg.logs.rootdir,
        "experiment_name": cfg.logs.experiment_name,
        "early_stage_logging_path": cfg.path.early_stage_logging_path,
        "late_stage_logging_path": cfg.path.late_stage_logging_path,
        "early_stage_naive_cf_path": cfg.path.early_stage_naive_cf_path,  # unused
        "late_stage_naive_cf_path": cfg.path.late_stage_naive_cf_path,  # unused
        "early_stage_moe_cf_path": cfg.path.early_stage_moe_cf_path,  # unused
        "early_stage_moe_model_selector_path": cfg.path.early_stage_moe_model_selector_path,  # unused
        "early_stage_quantile_cf_path": cfg.path.early_stage_quantile_cf_path,  # unused
        "early_stage_online_credit_assigned_pg_path": cfg.path.early_stage_online_credit_assigned_pg_path,
        "early_stage_online_vanilla_pg_path": cfg.path.early_stage_online_vanilla_pg_path,
        "early_stage_is_credit_assigned_pg_path": cfg.path.early_stage_is_credit_assigned_pg_path,
        "early_stage_is_vanilla_pg_path": cfg.path.early_stage_is_vanilla_pg_path,
        "early_stage_kernel_is_credit_assigned_pg_path": cfg.path.early_stage_kernel_is_credit_assigned_pg_path,
        "early_stage_kernel_vanilla_pg_path": cfg.path.early_stage_kernel_vanilla_pg_path,
        "logging_action_prob_model_path": cfg.path.logging_action_prob_model_path,
        "logging_marginal_model_path": cfg.path.logging_marginal_model_path,
    }
    process(conf)


if __name__ == "__main__":
    start = time()
    main()
    finish = time()
    print(f"Total runtime: {format_runtime(start, finish)}")
