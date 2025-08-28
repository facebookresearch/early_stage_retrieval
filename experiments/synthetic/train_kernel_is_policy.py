"""Train the policy via OPL (Kernel IS)."""

from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import hydra

import torch

from early_stage_retrieval.experiments.synthetic.function import (
    collect_logged_dataset,
    initialize_trainable_policy,
    initialize_uniform_policy,
    load_late_stage_cf_policy,
    load_logging_marginal_model,
    load_logging_policy,
    save_logs,
    setup_data_generation_process,
    train_kernel_is_pg_policy,
)
from early_stage_retrieval.experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
    reset_seed,
)
from manifold.clients.python import ManifoldClient
from omegaconf import DictConfig

# from .function import (
#    collect_logged_dataset,
#    initialize_trainable_policy,
#    initialize_uniform_policy,
#    load_logging_policy,
#    save_logs,
#    setup_data_generation_process,
#    train_is_pg_policy,
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
    data_size: int,
    dim_model_emb: int,
    kernel_vanilla_pg_lr: Union[int, float],
    kernel_credit_assigned_pg_lr: Union[int, float],
    kernel_bandwidth: Union[int, float],
    credit_assignment_type: str,
    n_epoch_logging: int,
    n_epochs_per_log: int,
    n_candidate_action_logging: int,
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
    early_stage_logging_path: str,
    late_stage_logging_path: str,
    late_stage_naive_cf_path: str,
    logging_marginal_model_path: str,
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

    if logging_type == "uniform":
        logging_early_stage_policy, logging_late_stage_policy = (
            initialize_uniform_policy(
                env=env,
                device=device,
                random_seed=base_random_seed,
            )
        )
    else:
        logging_early_stage_policy, logging_late_stage_policy = load_logging_policy(
            env=env,
            dim_model_emb=dim_model_emb,
            early_stage_logging_path=early_stage_logging_path,
            late_stage_logging_path=late_stage_logging_path,
            device=device,
            random_seed=base_random_seed,
        )

    reset_seed(random_seed)

    logged_dataset = collect_logged_dataset(
        env=env,
        logging_early_stage_policy=logging_early_stage_policy,
        logging_late_stage_policy=logging_late_stage_policy,
        is_deterministic_early_stage=(logging_type in ["practical", "deficient"]),
        is_deterministic_late_stage=(logging_type == "deficient"),
        n_candidate_action=n_candidate_action_logging,
        data_size=data_size,
    )

    late_stage_cf_policy = load_late_stage_cf_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        late_stage_naive_cf_path=late_stage_naive_cf_path,
        device=device,
        random_seed=random_seed,
    )
    logging_marginal_model = load_logging_marginal_model(
        env=env,
        base_model=late_stage_cf_policy.base_model,
        logging_marginal_model_path=logging_marginal_model_path,
        device=device,
        random_seed=random_seed,
    )

    is_early_stage_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=base_random_seed,
    )
    kernel_is_early_stage_policy, kernel_is_pg_training_logs = (
        train_kernel_is_pg_policy(
            env=env,
            logged_dataset=logged_dataset,
            logging_marginal_model=logging_marginal_model,
            kernel_bandwidth=kernel_bandwidth,
            early_stage_policy=is_early_stage_policy,
            early_stage_lr=kernel_credit_assigned_pg_lr
            if credit_assignment_type == "full"
            else kernel_vanilla_pg_lr,
            credit_assignment_type=credit_assignment_type,
            n_epoch=n_epoch_logging,
            n_epochs_per_log=n_epochs_per_log,
            n_candidate_action_train=n_candidate_action_train,
            n_candidate_action_eval=n_candidate_action_eval,
            device=device,
            random_seed=base_random_seed,
        )
    )
    save_logs(
        bucket=bucket,
        rootdir=rootdir,
        manifold_rootdir=manifold_rootdir,
        use_manifold=use_manifold,
        experiment_name=experiment_name,
        logging_type=logging_type,
        setting=setting,
        key_param=key_param,
        credit_assignment_type=credit_assignment_type,
        n_candidate_action_train=n_candidate_action_train,
        random_seed=base_random_seed,
        trained_kernel_is_pg_early_stage_policy=kernel_is_early_stage_policy,
        kernel_is_pg_training_logs=kernel_is_pg_training_logs,
    )


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param_name = conf["setting"]
    experiment_name = conf["experiment_name"]
    logging_type = conf["logging_type"]

    bucket = conf["bucket"]
    rootdir = conf["rootdir"]
    manifold_rootdir = conf["manifold_rootdir"]
    use_manifold = conf["use_manifold"]

    if (
        setting != "n_candidate_action_eval"
        and conf["n_candidate_action_train"] == "auto"
    ):
        conf_["n_candidate_action_train"] = conf["n_candidate_action_eval"]

    if experiment_name == "auto":
        experiment_name = setting

    rootdir = f"{rootdir}/{experiment_name}"
    manifold_rootdir = f"{manifold_rootdir}/{experiment_name}"

    if conf["early_stage_logging_path"] == "auto":
        conf_["early_stage_logging_path"] = f"{rootdir}/logging/early_stage_policy.pt"

    if conf["manifold_early_stage_logging_path"] == "auto":
        conf_["manifold_early_stage_logging_path"] = (
            f"{manifold_rootdir}/logging/early_stage_policy.pt"
        )

    if conf["late_stage_logging_path"] == "auto":
        conf_["late_stage_logging_path"] = f"{rootdir}/logging/late_stage_policy.pt"

    if conf["manifold_late_stage_logging_path"] == "auto":
        conf_["manifold_late_stage_logging_path"] = (
            f"{manifold_rootdir}/logging/late_stage_policy.pt"
        )

    if use_manifold:
        with ManifoldClient.get_client(bucket=bucket) as client:
            if not client.sync_exists(conf_["manifold_early_stage_logging_path"]):
                raise ValueError("manifold_early_stage_logging_path does not exist.")
            else:
                client.sync_get(
                    conf_["manifold_early_stage_logging_path"],
                    conf_["early_stage_logging_path"],
                )

            if not client.sync_exists(conf_["manifold_late_stage_logging_path"]):
                raise ValueError("manifold_late_stage_logging_path does not exist.")
            else:
                client.sync_get(
                    conf_["manifold_late_stage_logging_path"],
                    conf_["late_stage_logging_path"],
                )

    if not Path(conf_["early_stage_logging_path"]).exists():
        raise ValueError("early_stage_logging_path does not exist.")

    if not Path(conf_["late_stage_logging_path"]).exists():
        raise ValueError("late_stage_logging_path does not exist.")

    if setting in ["default", "n_candidate_action_eval"]:
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={conf_['random_seed']}"
            manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={conf_['random_seed']}"

            if conf["late_stage_naive_cf_path"] == "auto":
                conf_["late_stage_naive_cf_path"] = (
                    f"{rootdir_}/naive_cf/late_stage_policy.pt"
                )
            if conf["manifold_late_stage_naive_cf_path"] == "auto":
                conf_["manifold_late_stage_naive_cf_path"] = (
                    f"{manifold_rootdir_}/naive_cf/late_stage_policy.pt"
                )

            if conf["logging_marginal_model_path"] == "auto":
                conf_["logging_marginal_model_path"] = (
                    f"{rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                )
            if conf["manifold_logging_marginal_model_path"] == "auto":
                conf_["manifold_logging_marginal_model_path"] = (
                    f"{manifold_rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                )

            if use_manifold:
                with ManifoldClient.get_client(bucket=bucket) as client:
                    if not client.sync_exists(
                        conf_["manifold_late_stage_naive_cf_path"]
                    ):
                        raise ValueError(
                            "manifold_late_stage_naive_cf_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_["manifold_late_stage_naive_cf_path"],
                            conf_["late_stage_naive_cf_path"],
                        )

                    if not client.sync_exists(
                        conf_["manifold_logging_marginal_model_path"]
                    ):
                        raise ValueError(
                            "manifold_logging_marginal_model_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_["manifold_logging_marginal_model_path"],
                            conf_["logging_marginal_model_path"],
                        )

            if not Path(conf_["late_stage_naive_cf_path"]).exists():
                raise ValueError("late_stage_naive_cf_path does not exist.")

            if not Path(conf_["logging_marginal_model_path"]).exists():
                raise ValueError("logging_marginal_model_path does not exist.")

    else:
        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            for random_seed in range(conf["n_random_seed"]):
                conf_["random_seed"] = random_seed + conf["start_random_seed"]

                rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},seed={conf_['random_seed']}"
                manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},seed={conf_['random_seed']}"

                if conf["late_stage_naive_cf_path"] == "auto":
                    conf_["late_stage_naive_cf_path"] = (
                        f"{rootdir_}/naive_cf/late_stage_policy.pt"
                    )
                if conf["manifold_late_stage_naive_cf_path"] == "auto":
                    conf_["manifold_late_stage_naive_cf_path"] = (
                        f"{manifold_rootdir_}/naive_cf/late_stage_policy.pt"
                    )

                if conf["logging_marginal_model_path"] == "auto":
                    conf_["logging_marginal_model_path"] = (
                        f"{rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                    )
                if conf["manifold_logging_marginal_model_path"] == "auto":
                    conf_["manifold_logging_marginal_model_path"] = (
                        f"{manifold_rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                    )

                if use_manifold:
                    with ManifoldClient.get_client(bucket=bucket) as client:
                        if not client.sync_exists(
                            conf_["manifold_late_stage_naive_cf_path"]
                        ):
                            raise ValueError(
                                "manifold_late_stage_naive_cf_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_["manifold_late_stage_naive_cf_path"],
                                conf_["late_stage_naive_cf_path"],
                            )
                        if not client.sync_exists(
                            conf_["manifold_logging_marginal_model_path"]
                        ):
                            raise ValueError(
                                "manifold_logging_marginal_model_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_["manifold_logging_marginal_model_path"],
                                conf_["logging_marginal_model_path"],
                            )

                if not Path(conf_["late_stage_naive_cf_path"]).exists():
                    raise ValueError("late_stage_naive_cf_path does not exist.")

                if not Path(conf_["logging_marginal_model_path"]).exists():
                    raise ValueError("logging_marginal_model_path does not exist.")

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

                if setting != "n_candidate_action_eval":
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},seed={conf_['random_seed']}"
                else:
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={conf_['random_seed']}"

                if conf["late_stage_naive_cf_path"] == "auto":
                    conf_["late_stage_naive_cf_path"] = (
                        f"{rootdir_}/naive_cf/late_stage_policy.pt"
                    )
                if conf["manifold_late_stage_naive_cf_path"] == "auto":
                    conf_["manifold_late_stage_naive_cf_path"] = (
                        f"{manifold_rootdir_}/naive_cf/late_stage_policy.pt"
                    )

                if conf["logging_marginal_model_path"] == "auto":
                    conf_["logging_marginal_model_path"] = (
                        f"{rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                    )
                if conf["manifold_logging_marginal_model_path"] == "auto":
                    conf_["manifold_logging_marginal_model_path"] = (
                        f"{manifold_rootdir_}/logging_marginal_model/logging_marginal_model.pt"
                    )

                print(
                    f"Setting: {setting}, Key Param: {key_param}, Random Seed: {random_seed}/{conf['n_random_seed']}"
                )
                _process(**conf_)

    else:
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={conf_['random_seed']}"

            if conf["late_stage_naive_cf_path"] == "auto":
                conf_["late_stage_naive_cf_path"] = (
                    f"{rootdir_}/naive_cf/late_stage_policy.pt"
                )
            if conf["manifold_late_stage_naive_cf_path"] == "auto":
                conf_["manifold_late_stage_naive_cf_path"] = (
                    f"{manifold_rootdir_}/naive_cf/late_stage_policy.pt"
                )

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
