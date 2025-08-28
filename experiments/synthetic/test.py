"""File to test .py file execution on kernel and manifold."""

from pathlib import Path
from time import time
from typing import Any, Dict

import hydra

import torch

from early_stage_retrieval.experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
)

from early_stage_retrieval.synthetic.dataset import (
    SyntheticDataGenerator,
)
from manifold.clients.python import ManifoldClient
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

    bucket = conf["bucket"]
    rootdir = conf["rootdir"]
    manifold_rootdir = conf["manifold_rootdir"]
    use_manifold = conf["use_manifold"]

    rootdir = Path(f"{rootdir}/test")
    manifold_rootdir = f"{manifold_rootdir}/test"

    rootdir.mkdir(parents=True, exist_ok=True)

    if use_manifold:
        with ManifoldClient.get_client(bucket=bucket) as client:
            if not client.sync_exists(manifold_rootdir):
                client.sync_mkdir(manifold_rootdir, recursive=True)

    original_local_path = Path(rootdir / "original_tensor.pt")
    download_local_path = Path(rootdir / "downloaded_tensor.pt")
    upload_manifold_path = f"{manifold_rootdir}/uploaded_tensor.pt"

    torch.save(tmp_tensor, original_local_path)

    with ManifoldClient.get_client(bucket=bucket) as client:
        if not client.sync_exists(manifold_rootdir):
            raise ValueError("manifold_rootdir does not exist")

        client.sync_put(
            upload_manifold_path,
            str(original_local_path),
            predicate=ManifoldClient.Predicates.AllowOverwrite,
        )
        client.sync_get(
            upload_manifold_path,
            str(download_local_path),
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
