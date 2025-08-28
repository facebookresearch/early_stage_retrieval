"""Evaluate OPL-learned policies on synthetic simulation."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Union

import hydra
import pandas as pd

import torch

from early_stage_retrieval.experiments.synthetic.function import (
    evaluate_policy,
    initialize_optimal_policy,
    load_pg_policy,
    setup_data_generation_process,
)
from early_stage_retrieval.experiments.synthetic.utils import (
    assert_configuration,
    defaultdict_to_dict,
    format_runtime,
    reset_seed,
)
from manifold.clients.python import ManifoldClient
from omegaconf import DictConfig

# from .function import (
#     collect_logged_dataset,
#     evaluate_policy,
#     initialize_optimal_policy,
#     initialize_uniform_policy,
#     load_greedy_algorithm,
#     load_moe_cf_policy,
#     load_moe_model_selector,
#     load_naive_cf_policy,
#     optimize_moe_model_assignment,
#     setup_data_generation_process,
# )
# from .utils import assert_configuration, defaultdict_to_dict, format_runtime, reset_seed


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
    random_seed: int,
    early_stage_is_credit_assigned_pg_path: str,
    early_stage_is_vanilla_pg_path: str,
    early_stage_kernel_is_credit_assigned_pg_path: str,
    early_stage_kernel_is_vanilla_pg_path: str,
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

    _, optimal_late_stage_policy = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    early_stage_is_credit_assigned_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_is_credit_assigned_pg_path,
        device=device,
        random_seed=random_seed,
    )
    early_stage_is_vanilla_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_is_vanilla_pg_path,
        device=device,
        random_seed=random_seed,
    )
    early_stage_kernel_is_credit_assigned_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_kernel_is_credit_assigned_pg_path,
        device=device,
        random_seed=random_seed,
    )
    early_stage_kernel_is_vanilla_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_kernel_is_vanilla_pg_path,
        device=device,
        random_seed=random_seed,
    )

    algorithms = [
        "is-credit-assigned",
        "is-vanilla",
        "kernel-is-credit-assigned",
        "kernel-is-vanilla",
    ]
    early_stage_policies = [
        early_stage_is_credit_assigned_pg_policy,
        early_stage_is_vanilla_pg_policy,
        early_stage_kernel_is_credit_assigned_pg_policy,
        early_stage_kernel_is_vanilla_pg_policy,
    ]

    performance = {}
    for i, algo in enumerate(algorithms):
        performance[algo] = evaluate_policy(
            env=env,
            early_stage_policy=early_stage_policies[i],
            late_stage_policy=optimal_late_stage_policy,
            n_candidate_action=n_candidate_action_eval,
            is_deterministic_early_stage=True,
            is_deterministic_late_stage=True,
        )

    return performance


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
        if setting == "n_candidate_action_eval":
            experiment_name = "default"
        else:
            experiment_name = setting

    if setting == "default" or (
        setting == "n_candidate_action_eval"
        and conf["n_candidate_action_train"] != "auto"
    ):
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
            manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

            if conf["early_stage_is_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_is_credit_assigned_pg_path"] = (
                    f"{rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["early_stage_kernel_is_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_kernel_is_credit_assigned_pg_path"] = (
                    f"{rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_is_credit_assigned_pg_path"] == "auto":
                conf_["manifold_early_stage_is_credit_assigned_pg_path"] = (
                    f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_kernel_is_credit_assigned_pg_path"] == "auto":
                conf_["manifold_early_stage_kernel_is_credit_assigned_pg_path"] = (
                    f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if use_manifold:
                with ManifoldClient.get_client(bucket=bucket) as client:
                    if not client.sync_exists(
                        conf_["manifold_early_stage_is_credit_assigned_pg_path"]
                    ):
                        raise ValueError(
                            "manifold_early_stage_is_credit_assigned_pg_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_["manifold_early_stage_is_credit_assigned_pg_path"],
                            conf_["early_stage_is_credit_assigned_pg_path"],
                        )

                    if not client.sync_exists(
                        conf_["manifold_early_stage_kernel_is_credit_assigned_pg_path"]
                    ):
                        raise ValueError(
                            "manifold_early_stage_kernel_is_credit_assigned_pg_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_[
                                "manifold_early_stage_kernel_is_credit_assigned_pg_path"
                            ],
                            conf_["early_stage_kernel_is_credit_assigned_pg_path"],
                        )

            if not Path(conf_["early_stage_is_credit_assigned_pg_path"]).exists():
                raise ValueError(
                    "early_stage_is_credit_assigned_pg_path does not exist."
                )

            if not Path(
                conf_["early_stage_kernel_is_credit_assigned_pg_path"]
            ).exists():
                raise ValueError(
                    "early_stage_kernel_is_credit_assigned_pg_path does not exist."
                )

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
            manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

            if conf["early_stage_is_vanilla_pg_path"] == "auto":
                conf_["early_stage_is_vanilla_pg_path"] = (
                    f"{rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["early_stage_kernel_is_vanilla_pg_path"] == "auto":
                conf_["early_stage_kernel_is_vanilla_pg_path"] = (
                    f"{rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_is_vanilla_pg_path"] == "auto":
                conf_["manifold_early_stage_is_vanilla_pg_path"] = (
                    f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_kernel_is_vanilla_pg_path"] == "auto":
                conf_["manifold_early_stage_kernel_is_vanilla_pg_path"] = (
                    f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if use_manifold:
                with ManifoldClient.get_client(bucket=bucket) as client:
                    if not client.sync_exists(
                        conf_["manifold_early_stage_is_vanilla_pg_path"]
                    ):
                        raise ValueError(
                            "manifold_early_stage_is_vanilla_pg_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_["manifold_early_stage_is_vanilla_pg_path"],
                            conf_["early_stage_is_vanilla_pg_path"],
                        )

                    if not client.sync_exists(
                        conf_["manifold_early_stage_kernel_is_vanilla_pg_path"]
                    ):
                        raise ValueError(
                            "manifold_early_stage_kernel_is_vanilla_pg_path does not exist."
                        )
                    else:
                        client.sync_get(
                            conf_["manifold_early_stage_kernel_is_vanilla_pg_path"],
                            conf_["early_stage_kernel_is_vanilla_pg_path"],
                        )

            if not Path(conf_["early_stage_is_vanilla_pg_path"]).exists():
                raise ValueError("early_stage_is_vanilla_pg_path does not exist.")

            if not Path(conf_["early_stage_kernel_is_vanilla_pg_path"]).exists():
                raise ValueError(
                    "early_stage_kernel_is_vanilla_pg_path does not exist."
                )

    else:
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

                rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
                manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

                if conf["early_stage_is_credit_assigned_pg_path"] == "auto":
                    conf_["early_stage_is_credit_assigned_pg_path"] = (
                        f"{rootdir_}/IS/early_stage_policy.pt"
                    )

                if conf["early_stage_kernel_is_credit_assigned_pg_path"] == "auto":
                    conf_["early_stage_kernel_is_credit_assigned_pg_path"] = (
                        f"{rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if conf["manifold_early_stage_is_credit_assigned_pg_path"] == "auto":
                    conf_["manifold_early_stage_is_credit_assigned_pg_path"] = (
                        f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                    )

                if (
                    conf["manifold_early_stage_kernel_is_credit_assigned_pg_path"]
                    == "auto"
                ):
                    conf_["manifold_early_stage_kernel_is_credit_assigned_pg_path"] = (
                        f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if use_manifold:
                    with ManifoldClient.get_client(bucket=bucket) as client:
                        if not client.sync_exists(
                            conf_["manifold_early_stage_is_credit_assigned_pg_path"]
                        ):
                            raise ValueError(
                                "manifold_early_stage_is_credit_assigned_pg_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_[
                                    "manifold_early_stage_is_credit_assigned_pg_path"
                                ],
                                conf_["early_stage_is_credit_assigned_pg_path"],
                            )

                        if not client.sync_exists(
                            conf_[
                                "manifold_early_stage_kernel_is_credit_assigned_pg_path"
                            ]
                        ):
                            raise ValueError(
                                "manifold_early_stage_kernel_is_credit_assigned_pg_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_[
                                    "manifold_early_stage_kernel_is_credit_assigned_pg_path"
                                ],
                                conf_["early_stage_kernel_is_credit_assigned_pg_path"],
                            )

                if not Path(conf_["early_stage_is_credit_assigned_pg_path"]).exists():
                    raise ValueError(
                        "early_stage_is_credit_assigned_pg_path does not exist."
                    )

                if not Path(
                    conf_["early_stage_kernel_is_credit_assigned_pg_path"]
                ).exists():
                    raise ValueError(
                        "early_stage_kernel_is_credit_assigned_pg_path does not exist."
                    )

                rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
                manifold_rootdir_ = f"{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

                if conf["early_stage_is_vanilla_pg_path"] == "auto":
                    conf_["early_stage_is_vanilla_pg_path"] = (
                        f"{rootdir_}/IS/early_stage_policy.pt"
                    )

                if conf["early_stage_kernel_is_vanilla_pg_path"] == "auto":
                    conf_["early_stage_kernel_is_vanilla_pg_path"] = (
                        f"{rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if conf["manifold_early_stage_is_vanilla_pg_path"] == "auto":
                    conf_["manifold_early_stage_is_vanilla_pg_path"] = (
                        f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                    )

                if conf["manifold_early_stage_kernel_is_vanilla_pg_path"] == "auto":
                    conf_["manifold_early_stage_kernel_is_vanilla_pg_path"] = (
                        f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if use_manifold:
                    with ManifoldClient.get_client(bucket=bucket) as client:
                        if not client.sync_exists(
                            conf_["manifold_early_stage_is_vanilla_pg_path"]
                        ):
                            raise ValueError(
                                "manifold_early_stage_is_vanilla_pg_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_["manifold_early_stage_is_vanilla_pg_path"],
                                conf_["early_stage_is_vanilla_pg_path"],
                            )

                        if not client.sync_exists(
                            conf_["manifold_early_stage_kernel_is_vanilla_pg_path"]
                        ):
                            raise ValueError(
                                "manifold_early_stage_kernel_is_vanilla_pg_path does not exist."
                            )
                        else:
                            client.sync_get(
                                conf_["manifold_early_stage_kernel_is_vanilla_pg_path"],
                                conf_["early_stage_kernel_is_vanilla_pg_path"],
                            )

                if not Path(conf_["early_stage_is_vanilla_pg_path"]).exists():
                    raise ValueError("early_stage_is_vanilla_pg_path does not exist.")

                if not Path(conf_["early_stage_kernel_is_vanilla_pg_path"]).exists():
                    raise ValueError(
                        "early_stage_kernel_is_vanilla_pg_path does not exist."
                    )

    performance_dict = defaultdict(list)

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
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
                else:
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

                if conf["early_stage_is_credit_assigned_pg_path"] == "auto":
                    conf_["early_stage_is_credit_assigned_pg_path"] = (
                        f"{rootdir_}/IS/early_stage_policy.pt"
                    )

                if conf["manifold_early_stage_is_credit_assigned_pg_path"] == "auto":
                    conf_["manifold_early_stage_is_credit_assigned_pg_path"] = (
                        f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                    )

                if (
                    conf["manifold_early_stage_kernel_is_credit_assigned_pg_path"]
                    == "auto"
                ):
                    conf_["manifold_early_stage_kernel_is_credit_assigned_pg_path"] = (
                        f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if setting != "n_candidate_action_eval":
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"
                else:
                    rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

                if conf["early_stage_is_vanilla_pg_path"] == "auto":
                    conf_["early_stage_is_vanilla_pg_path"] = (
                        f"{rootdir_}/IS/early_stage_policy.pt"
                    )

                if conf["early_stage_kernel_is_vanilla_pg_path"] == "auto":
                    conf_["early_stage_kernel_is_vanilla_pg_path"] = (
                        f"{rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                if conf["manifold_early_stage_kernel_is_vanilla_pg_path"] == "auto":
                    conf_["manifold_early_stage_kernel_is_vanilla_pg_path"] = (
                        f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                    )

                print(
                    f"Setting: {setting}, Key Param: {key_param}, Random Seed: {random_seed}/{conf['n_random_seed']}"
                )

                performance_ = _process(**conf_)
                performance_dict[key_param_name].append(key_param)
                performance_dict["random_seed"].append(random_seed)

                for key, value in performance_.items():
                    performance_dict[key].append(value)

    else:
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'full'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

            if conf["early_stage_is_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_is_credit_assigned_pg_path"] = (
                    f"{rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["early_stage_kernel_is_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_kernel_is_credit_assigned_pg_path"] = (
                    f"{rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_is_credit_assigned_pg_path"] == "auto":
                conf_["manifold_early_stage_is_credit_assigned_pg_path"] = (
                    f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_kernel_is_credit_assigned_pg_path"] == "auto":
                conf_["manifold_early_stage_kernel_is_credit_assigned_pg_path"] = (
                    f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                )

            rootdir_ = f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={'partial'},n_candidate_action_train={conf_['n_candidate_action_train']},seed={conf_['random_seed']}"

            if conf["early_stage_is_vanilla_pg_path"] == "auto":
                conf_["early_stage_is_vanilla_pg_path"] = (
                    f"{rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["early_stage_kernel_is_vanilla_pg_path"] == "auto":
                conf_["early_stage_kernel_is_vanilla_pg_path"] = (
                    f"{rootdir_}/kernelIS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_is_vanilla_pg_path"] == "auto":
                conf_["manifold_early_stage_is_vanilla_pg_path"] = (
                    f"{manifold_rootdir_}/IS/early_stage_policy.pt"
                )

            if conf["manifold_early_stage_kernel_is_vanilla_pg_path"] == "auto":
                conf_["manifold_early_stage_kernel_is_vanilla_pg_path"] = (
                    f"{manifold_rootdir_}/kernelIS/early_stage_policy.pt"
                )

            print(
                f"Setting: {setting}, Key Param: {'None'}, Random Seed: {random_seed}/{conf['n_random_seed']}"
            )

            performance_ = _process(**conf_)
            performance_dict["setting"].append("default")
            performance_dict["random_seed"].append(random_seed)

            for key, value in performance_.items():
                performance_dict[key].append(value)

    performance_dict = defaultdict_to_dict(performance_dict)
    df = pd.DataFrame(performance_dict)

    Path(f"{rootdir}/{experiment_name}").mkdir(parents=True, exist_ok=True)
    df_path = f"{rootdir}/{experiment_name}/offline_pg_performance_{logging_type}.csv"
    df.to_csv(df_path, index=False)

    if use_manifold:
        with ManifoldClient.get_client(bucket=bucket) as client:
            if not client.sync_exists(manifold_rootdir):
                client.sync_mkdir(manifold_rootdir, recursive=True)

            manifold_df_path = f"{manifold_rootdir}/{experiment_name}/offline_pg_performance_{logging_type}.csv"
            client.sync_put(
                manifold_df_path,
                df_path,
                predicate=ManifoldClient.Predicates.AllowOverwrite,
            )


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
        "early_stage_logging_path": cfg.path.early_stage_logging_path,
        "late_stage_logging_path": cfg.path.late_stage_logging_path,
        "early_stage_naive_cf_path": cfg.path.early_stage_naive_cf_path,
        "late_stage_naive_cf_path": cfg.path.late_stage_naive_cf_path,
        "early_stage_moe_cf_path": cfg.path.early_stage_moe_cf_path,
        "early_stage_moe_model_selector_path": cfg.path.early_stage_moe_model_selector_path,
        "early_stage_quantile_cf_path": cfg.path.early_stage_quantile_cf_path,
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
