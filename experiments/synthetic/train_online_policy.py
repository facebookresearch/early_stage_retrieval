"""Train the policy via Online PG."""

from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import hydra

import torch

from early_stage_retrieval.experiments.synthetic.function import (
    initialize_trainable_policy,
    save_logs,
    setup_data_generation_process,
    train_online_pg_policy,
)
from early_stage_retrieval.experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
    reset_seed,
)
from omegaconf import DictConfig

# from .function import (
#     initialize_trainable_policy,
#     initialize_uniform_policy,
#     save_logs,
#     setup_data_generation_process,
#     train_online_pg_policy,
# )
# from .utils import assert_configuration, format_runtime, reset_seed


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
    dim_model_emb: int,
    online_vanilla_pg_lr: Union[int, float],
    online_credit_assigned_pg_lr: Union[int, float],
    credit_assignment_type: str,
    n_epoch_logging: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    bucket: str,
    rootdir: str,
    manifold_rootdir: str,
    use_manifold: bool,
    experiment_name: str,
    logging_type: str,
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

    reset_seed(random_seed)

    online_early_stage_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=base_random_seed,
    )
    online_early_stage_policy, online_pg_training_logs = train_online_pg_policy(
        env=env,
        early_stage_policy=online_early_stage_policy,
        early_stage_lr=online_credit_assigned_pg_lr
        if credit_assignment_type == "full"
        else online_vanilla_pg_lr,
        credit_assignment_type=credit_assignment_type,
        n_epoch=n_epoch_logging,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action_train=n_candidate_action_train,
        n_candidate_action_eval=n_candidate_action_eval,
        device=device,
        random_seed=base_random_seed,
    )
    save_logs(
        bucket=bucket,
        rootdir=rootdir,
        manifold_rootdir=manifold_rootdir,
        use_manifold=use_manifold,
        experiment_name=experiment_name,
        logging_type=logging_type,
        credit_assignment_type=credit_assignment_type,
        n_candidate_action_train=n_candidate_action_train,
        setting=setting,
        key_param=key_param,
        random_seed=base_random_seed,
        trained_online_pg_early_stage_policy=online_early_stage_policy,
        online_pg_training_logs=online_pg_training_logs,
    )


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param = conf["setting"]

    if (
        setting != "n_candidate_action_eval"
        and conf["n_candidate_action_train"] == "auto"
    ):
        conf_["n_candidate_action_train"] = conf["n_candidate_action_eval"]

    if setting != "default":
        key_param_name = conf["setting"]

        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            if (
                setting == "n_candidate_action_eval"
                and conf["n_candidate_action_train"] == "auto"
            ):
                conf_["n_candidate_action_train"] = key_param

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
        "bucket": cfg.logs.bucket,
        "rootdir": cfg.logs.rootdir,
        "manifold_rootdir": cfg.logs.manifold_rootdir,
        "experiment_name": cfg.logs.experiment_name,
        "use_manifold": cfg.logs.use_manifold,
        "early_stage_logging_path": cfg.path.early_stage_logging_path,  # unused
        "late_stage_logging_path": cfg.path.late_stage_logging_path,  # unused
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
        "manifold_early_stage_logging_path": cfg.path.manifold_early_stage_logging_path,
        "manifold_late_stage_logging_path": cfg.path.manifold_late_stage_logging_path,
        "manifold_early_stage_naive_cf_path": cfg.path.manifold_early_stage_naive_cf_path,
        "manifold_late_stage_naive_cf_path": cfg.path.manifold_late_stage_naive_cf_path,
        "manifold_early_stage_moe_cf_path": cfg.path.manifold_early_stage_moe_cf_path,
        "manifold_early_stage_moe_model_selector_path": cfg.path.manifold_early_stage_moe_model_selector_path,
        "manifold_early_stage_quantile_cf_path": cfg.path.manifold_early_stage_quantile_cf_path,
        "manifold_early_stage_online_credit_assigned_pg_path": cfg.path.early_stage_online_credit_assigned_pg_path,
        "manifold_early_stage_online_vanilla_pg_path": cfg.path.early_stage_online_vanilla_pg_path,
        "manifold_early_stage_is_credit_assigned_pg_path": cfg.path.early_stage_is_credit_assigned_pg_path,
        "manifold_early_stage_is_vanilla_pg_path": cfg.path.early_stage_is_vanilla_pg_path,
        "manifold_early_stage_kernel_is_credit_assigned_pg_path": cfg.path.early_stage_kernel_is_credit_assigned_pg_path,
        "manifold_early_stage_kernel_vanilla_pg_path": cfg.path.early_stage_kernel_vanilla_pg_path,
        "manifold_logging_action_prob_model_path": cfg.path.manifold_logging_action_prob_model_path,
        "manifold_logging_marginal_model_path": cfg.path.manifold_logging_marginal_model_path,
    }
    process(conf)


if __name__ == "__main__":
    start = time()
    main()
    finish = time()
    print(f"Total runtime: {format_runtime(start, finish)}")
