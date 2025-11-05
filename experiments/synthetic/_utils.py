"""Useful tools for teh experiments."""

import random
from typing import Union

import torch

from omegaconf import DictConfig


def format_runtime(start: Union[int, float], finish: Union[int, float]) -> str:
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime) // 60 % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"


def defaultdict_to_dict(d):
    if isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def reset_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def assert_configuration(cfg: DictConfig) -> None:
    setting = cfg.setting.setting
    assert setting in [
        "default",
        "data_size",
        "n_action",
        "n_candidate_action_eval",
        "n_latent",
        "n_moe_model",
        "logging_type",
    ]

    data_size = cfg.setting.data_size
    if setting != "data_size":
        assert data_size in [10000, 50000, 100000, 500000, 1000000]
    else:
        for value in data_size:
            assert value in [10000, 50000, 100000, 500000, 1000000]

    n_action = cfg.setting.n_action
    if setting != "n_action":
        assert n_action in [1000, 2000, 5000, 10000]
    else:
        for value in n_action:
            assert value in [1000, 2000, 5000, 10000]

    n_candidate_action_eval = cfg.setting.n_candidate_action_eval
    if setting != "n_candidate_action_eval":
        assert n_candidate_action_eval in [10, 20, 50, 100]
    else:
        for value in n_candidate_action_eval:
            assert value in [10, 20, 50, 100]

    n_output_action = cfg.setting.n_output_action
    assert n_output_action in [1, 2, 5, 10]

    n_candidate_action_logging = cfg.setting.n_candidate_action_logging
    assert n_candidate_action_logging in [100, 200, 500, 1000]

    n_candidate_action_train = cfg.setting.n_candidate_action_train
    if not n_candidate_action_train == "auto":
        assert n_candidate_action_train in [10, 20, 50, 100]

    n_user = cfg.setting.n_user
    assert n_user in [100, 1000, 10000]

    n_latent = cfg.setting.n_latent
    assert n_latent in [1, 5, 10, 50, 100]

    dim_context = cfg.setting.dim_context
    assert dim_context in [5, 10]

    dim_action_emb = cfg.setting.dim_action_emb
    assert dim_action_emb in [5, 10]

    reward_scaler = cfg.setting.reward_scaler
    assert reward_scaler >= 1

    logging_type = cfg.setting.logging_type
    assert logging_type in ["uniform", "skewed", "practical", "deficient"]

    device = cfg.setting.device
    assert device in ["cpu", "cuda"]

    n_random_seed = cfg.setting.n_random_seed
    assert n_random_seed >= 1

    start_random_seed = cfg.setting.start_random_seed
    assert start_random_seed >= 0

    base_random_seed = cfg.setting.base_random_seed
    assert base_random_seed >= 0

    dim_model_emb = cfg.model.dim_model_emb
    assert dim_model_emb in [5, 10]

    n_moe_model = cfg.model.n_moe_model
    assert n_moe_model in [1, 2, 5, 10]

    early_stage_logging_lr = cfg.model.early_stage_logging_lr
    assert early_stage_logging_lr >= 0

    late_stage_logging_lr = cfg.model.late_stage_logging_lr
    assert late_stage_logging_lr >= 0

    early_stage_naive_cf_lr = cfg.model.early_stage_naive_cf_lr
    assert early_stage_naive_cf_lr >= 0

    early_stage_moe_cf_lr = cfg.model.early_stage_moe_cf_lr
    assert early_stage_moe_cf_lr >= 0

    early_stage_moe_selector_lr = cfg.model.early_stage_moe_selector_lr
    assert early_stage_moe_selector_lr >= 0

    quantile_cf_lr = cfg.model.quantile_cf_lr
    assert quantile_cf_lr >= 0

    late_stage_neural_lr = cfg.model.late_stage_neural_lr
    assert late_stage_neural_lr >= 0

    online_vanilla_pg_lr = cfg.model.online_vanilla_pg_lr
    assert online_vanilla_pg_lr >= 0

    online_credit_assigned_pg_lr = cfg.model.online_credit_assigned_pg_lr
    assert online_credit_assigned_pg_lr >= 0

    is_vanilla_pg_lr = cfg.model.is_vanilla_pg_lr
    assert is_vanilla_pg_lr >= 0

    is_credit_assigned_pg_lr = cfg.model.is_credit_assigned_pg_lr
    assert is_credit_assigned_pg_lr >= 0

    kernel_vanilla_pg_lr = cfg.model.kernel_vanilla_pg_lr
    assert kernel_vanilla_pg_lr >= 0

    kernel_creedit_assigned_pg_lr = cfg.model.kernel_creedit_assigned_pg_lr
    assert kernel_creedit_assigned_pg_lr >= 0

    logging_action_prob_model_lr = cfg.model.logging_action_prob_model_lr
    assert logging_action_prob_model_lr >= 0

    logging_marginal_model_lr = cfg.model.logging_marginal_model_lr
    assert logging_marginal_model_lr >= 0

    n_epoch = cfg.model.n_epoch
    assert n_epoch >= 1

    n_epoch_regression = cfg.model.n_epoch_regression
    assert n_epoch_regression >= 1

    n_epoch_logging = cfg.model.n_epoch_logging
    assert n_epoch_logging >= 1

    n_steps_per_epoch = cfg.model.n_steps_per_epoch
    assert n_steps_per_epoch >= 1

    n_epochs_per_log = cfg.model.n_epochs_per_log
    assert n_epochs_per_log >= 1

    kernel_bandwidth = cfg.model.kernel_bandwidth
    assert kernel_bandwidth > 0

    credit_assignment_type = cfg.model.credit_assignment_type
    assert credit_assignment_type in ["full", "partial", "none"]

    early_stage_logging_path = cfg.path.early_stage_logging_path
    if early_stage_logging_path != "auto":
        assert early_stage_logging_path.endswith(".pt")

    late_stage_logging_path = cfg.path.late_stage_logging_path
    if late_stage_logging_path != "auto":
        assert late_stage_logging_path.endswith(".pt")

    early_stage_naive_cf_path = cfg.path.early_stage_naive_cf_path
    if early_stage_naive_cf_path != "auto":
        assert early_stage_naive_cf_path.endswith(".pt")

    late_stage_naive_cf_path = cfg.path.late_stage_naive_cf_path
    if late_stage_naive_cf_path != "auto":
        assert late_stage_naive_cf_path.endswith(".pt")

    early_stage_moe_cf_path = cfg.path.early_stage_moe_cf_path
    if early_stage_moe_cf_path != "auto":
        assert early_stage_moe_cf_path.endswith(".pt")

    early_stage_moe_model_selector_path = cfg.path.early_stage_moe_model_selector_path
    if early_stage_moe_model_selector_path != "auto":
        assert early_stage_moe_model_selector_path.endswith(".pt")

    early_stage_quantile_cf_path = cfg.path.early_stage_quantile_cf_path
    if early_stage_quantile_cf_path != "auto":
        assert early_stage_quantile_cf_path.endswith(".pt")

    early_stage_online_credit_assigned_pg_path = (
        cfg.path.early_stage_online_credit_assigned_pg_path
    )
    if early_stage_online_credit_assigned_pg_path != "auto":
        assert early_stage_online_credit_assigned_pg_path.endswith(".pt")

    early_stage_online_vanilla_pg_path = cfg.path.early_stage_online_vanilla_pg_path
    if early_stage_online_vanilla_pg_path != "auto":
        assert early_stage_online_vanilla_pg_path.endswith(".pt")

    early_stage_is_credit_assigned_pg_path = (
        cfg.path.early_stage_is_credit_assigned_pg_path
    )
    if early_stage_is_credit_assigned_pg_path != "auto":
        assert early_stage_is_credit_assigned_pg_path.endswith(".pt")

    early_stage_is_vanilla_pg_path = cfg.path.early_stage_is_vanilla_pg_path
    if early_stage_is_vanilla_pg_path != "auto":
        assert early_stage_is_vanilla_pg_path.endswith(".pt")

    early_stage_kernel_is_credit_assigned_pg_path = (
        cfg.path.early_stage_kernel_is_credit_assigned_pg_path
    )
    if early_stage_kernel_is_credit_assigned_pg_path != "auto":
        assert early_stage_kernel_is_credit_assigned_pg_path.endswith(".pt")

    logging_action_prob_model_lr_path = cfg.path.logging_action_prob_model_lr_path
    if logging_action_prob_model_lr_path != "auto":
        assert logging_action_prob_model_lr_path.endswith(".pt")

    logging_marginal_model_lr_path = cfg.path.logging_marginal_model_lr_path
    if logging_marginal_model_lr_path != "auto":
        assert logging_marginal_model_lr_path.endswith(".pt")
