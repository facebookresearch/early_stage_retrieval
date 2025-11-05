"""Functions called in the experiment pyfiles."""

from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from synthetic.dataset import (
    LoggedDataset,
    SyntheticDataGenerator,
    VectorialContextSampler,
    VectorialLatentSampler,
    VectorialRewardModel,
)
from synthetic.learner import (
    ActionProbModel,
    ActionProbRegressor,
    CollaborativeFilteringLearner,
    CollaborativeFilteringQuantileLearner,
    ExpertSelectionModel,
    gaussian_kernel,
    ImportanceSamplingLearner,
    KernelDensityModel,
    KernelDensityRegressor,
    KernelImportanceSamplingLearner,
    OnlinePolicyLearner,
)
from synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
    BaselineEarlyStagePolicy,
    BaselineLateStagePolicy,
    EarlyStageTwoTowerModel,
    EarlyStageTwoTowerQuantileModel,
    GreedySubsetEarlyStagePolicy,
    LateStageNeuralModel,
    OptimalEarlyStagePolicy,
    OptimalLateStagePolicy,
    UniformEarlyStagePolicy,
    UniformLateStagePolicy,
    VectorialActionSet,
)

def setup_data_generation_process(
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    device: torch.device,
    random_seed: int,
) -> SyntheticDataGenerator:
    context_sampler = VectorialContextSampler(
        is_discrete=True,
        n_discrete_context=n_user,
        dim_context=dim_context,
        device=device,
        random_seed=random_seed,
    )
    action_set = VectorialActionSet(
        n_action=n_action,
        dim_action_emb=dim_action_emb,
        device=device,
        random_seed=random_seed,
    )
    latent_sampler = VectorialLatentSampler(
        is_discrete=True,
        n_discrete_latent=n_latent,
        dim_context=context_sampler.dim_context,
        dim_action_emb=action_set.dim_action_emb,
        device=device,
        random_seed=random_seed,
    )
    reward_model = VectorialRewardModel(
        context_sampler=context_sampler,
        action_set=action_set,
        n_output_action=n_output_action,
        reward_scaler=reward_scaler,
        device=device,
        random_seed=random_seed,
    )
    datagen = SyntheticDataGenerator(
        action_set=action_set,
        context_sampler=context_sampler,
        latent_sampler=latent_sampler,
        reward_model=reward_model,
    )
    return datagen


def collect_logged_dataset(
    env: SyntheticDataGenerator,
    logging_early_stage_policy: BaseEarlyStagePolicy,
    logging_late_stage_policy: BaseLateStagePolicy,
    is_deterministic_early_stage: bool,
    is_deterministic_late_stage: bool,
    n_candidate_action: int,
    data_size: int,
) -> LoggedDataset:
    logged_dataset = env.sample_dataset(
        early_stage_policy=logging_early_stage_policy,
        late_stage_policy=logging_late_stage_policy,
        is_deterministic_early_stage=is_deterministic_early_stage,
        is_deterministic_late_stage=is_deterministic_late_stage,
        n_candidate_action=n_candidate_action,
        n_samples=data_size,
    )
    return logged_dataset


def evaluate_policy(
    env: SyntheticDataGenerator,
    early_stage_policy: BaseEarlyStagePolicy,
    late_stage_policy: BaseLateStagePolicy,
    is_deterministic_early_stage: bool,
    is_deterministic_late_stage: bool,
    n_candidate_action: int,
    model_assignment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    online_performance = env.evaluate_policy_online(
        early_stage_policy=early_stage_policy,
        late_stage_policy=late_stage_policy,
        is_deterministic_early_stage=is_deterministic_early_stage,
        is_deterministic_late_stage=is_deterministic_late_stage,
        n_candidate_action=n_candidate_action,
        n_candidate_per_model=model_assignment,
    ).item()
    return online_performance


def train_early_stage_and_late_stage_with_cf(
    env: SyntheticDataGenerator,
    logged_dataset: LoggedDataset,
    early_stage_policy: BaselineEarlyStagePolicy,
    late_stage_policy: BaselineLateStagePolicy,
    early_stage_lr: float,
    late_stage_lr: float,
    model_selector_lr: float,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineEarlyStagePolicy,
    BaselineLateStagePolicy,
    Optional[ExpertSelectionModel],
    Dict[str, torch.Tensor],
]:
    cf_learner = CollaborativeFilteringLearner(
        early_stage_policy=early_stage_policy,
        late_stage_policy=late_stage_policy,
        model_selector=ExpertSelectionModel(
            n_latent=env.latent_sampler.n_discrete_latent,
            n_model=early_stage_policy.base_model.n_model,
        ),
        early_stage_optimizer_kwargs={
            "lr": early_stage_lr,
        },
        late_stage_optimizer_kwargs={
            "lr": late_stage_lr,
        },
        model_selector_optimizer_kwargs={
            "lr": model_selector_lr,
        },
        env=env,
        device=device,
        random_seed=random_seed,
    )
    (
        trained_early_stage_policy,
        trained_late_stage_policy,
        trained_model_selector,
        cf_training_logs,
    ) = cf_learner.train_early_and_late_stage_policies_offline(
        dataset=logged_dataset,
        loss_type="mse",
        n_epoch=n_epoch,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action=n_candidate_action_eval,
        is_deterministic_early_stage_eval=True,
        is_deterministic_late_stage_eval=True,
        make_copy=False,  #
        return_training_logs=True,
        patience=torch.inf,
        random_seed=random_seed,
    )
    return (
        trained_early_stage_policy,
        trained_late_stage_policy,
        trained_model_selector,
        cf_training_logs,
    )


def train_early_stage_with_cf(
    env: SyntheticDataGenerator,
    logged_dataset: LoggedDataset,
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    model_selector_lr: float,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Optional[EarlyStageTwoTowerQuantileModel],
    Dict[str, torch.Tensor],
]:
    _, late_stage_policy_optimal = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )

    cf_learner = CollaborativeFilteringLearner(
        early_stage_policy=early_stage_policy,
        late_stage_policy=late_stage_policy_optimal,
        model_selector=ExpertSelectionModel(
            n_latent=env.latent_sampler.n_discrete_latent,
            n_model=early_stage_policy.base_model.n_model,
        ),
        is_model_free_late_stage_policy=True,
        early_stage_optimizer_kwargs={
            "lr": early_stage_lr,
        },
        model_selector_optimizer_kwargs={
            "lr": model_selector_lr,
        },
        env=env,
        device=device,
        random_seed=random_seed,
    )
    (
        trained_early_stage_policy,
        trained_model_selector,
        cf_training_logs,
    ) = cf_learner.train_early_stage_policy_offline(
        dataset=logged_dataset,
        loss_type="mse",
        n_epoch=n_epoch,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action=n_candidate_action_eval,
        is_deterministic_early_stage_eval=True,
        is_deterministic_late_stage_eval=True,
        make_copy=False,  #
        return_training_logs=True,
        patience=torch.inf,
        random_seed=random_seed,
    )
    return (
        trained_early_stage_policy,
        trained_model_selector,
        cf_training_logs,
    )


def train_late_stage_with_cf(
    env: SyntheticDataGenerator,
    logged_dataset: LoggedDataset,
    late_stage_policy: BaselineLateStagePolicy,
    late_stage_lr: float,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineLateStagePolicy,
    Dict[str, torch.Tensor],
]:
    early_stage_policy_optimal, _ = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )

    cf_learner = CollaborativeFilteringLearner(
        early_stage_policy=early_stage_policy_optimal,
        late_stage_policy=late_stage_policy,
        is_model_free_early_stage_policy=True,
        late_stage_optimizer_kwargs={
            "lr": late_stage_lr,
        },
        env=env,
        device=device,
        random_seed=random_seed,
    )
    (
        trained_late_stage_policy,
        cf_training_logs,
    ) = cf_learner.train_late_stage_policy_offline(
        dataset=logged_dataset,
        loss_type="mse",
        n_epoch=n_epoch,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action=n_candidate_action_eval,
        is_deterministic_early_stage_eval=True,
        is_deterministic_late_stage_eval=True,
        make_copy=False,  #
        return_training_logs=True,
        patience=torch.inf,
        random_seed=random_seed,
    )
    return (
        trained_late_stage_policy,
        cf_training_logs,
    )


def train_quantile_model_with_cf(
    logged_dataset: LoggedDataset,
    quantile_cf_model: EarlyStageTwoTowerQuantileModel,
    quantile_cf_lr: float,
    n_epoch: int,
    n_epochs_per_log: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    EarlyStageTwoTowerQuantileModel,
    Dict[str, torch.Tensor],
]:
    quantile_cf_learner = CollaborativeFilteringQuantileLearner(
        model=quantile_cf_model,
        optimizer_kwargs={
            "lr": quantile_cf_lr,
        },
        device=device,
        random_seed=random_seed,
    )
    trained_quantile_cf_model, quantile_cf_training_logs = (
        quantile_cf_learner.train_model_offline(
            dataset=logged_dataset,
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            random_seed=random_seed,
        )
    )
    return trained_quantile_cf_model, quantile_cf_training_logs


def train_logging_action_prob_model(
    logged_dataset: LoggedDataset,
    logging_early_stage_policy: BaselineEarlyStagePolicy,
    logging_late_stage_policy: BaselineLateStagePolicy,
    is_deterministic_early_stage_logging: bool,
    is_deterministic_late_stage_logging: bool,
    action_prob_model: ActionProbModel,
    action_prob_model_lr: float,
    n_candidate_action_logging: int,
    n_epoch: int,
    n_epochs_per_log: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    ActionProbModel,
    Dict[str, torch.Tensor],
]:
    action_prob_model_learner = ActionProbRegressor(
        model=action_prob_model,
        optimizer_kwargs={"lr": action_prob_model_lr},
        device=device,
        random_seed=random_seed,
    )
    trained_logging_action_prob_model, logging_action_prob_model_training_logs = (
        action_prob_model_learner.train_model_offline(
            dataset=logged_dataset,
            early_stage_policy=logging_early_stage_policy,
            late_stage_policy=logging_late_stage_policy,
            is_deterministic_early_stage=is_deterministic_early_stage_logging,
            is_deterministic_late_stage=is_deterministic_late_stage_logging,
            n_candidate_action=n_candidate_action_logging,
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            random_seed=random_seed,
        )
    )
    return trained_logging_action_prob_model, logging_action_prob_model_training_logs


def train_logging_marginal_model(
    logged_dataset: LoggedDataset,
    logging_early_stage_policy: BaselineEarlyStagePolicy,
    logging_late_stage_policy: BaselineLateStagePolicy,
    is_deterministic_early_stage_logging: bool,
    is_deterministic_late_stage_logging: bool,
    marginal_model: KernelDensityModel,
    marginal_model_lr: float,
    kernel_bandwidth: Union[int, float],
    n_candidate_action_logging: int,
    n_epoch: int,
    n_epochs_per_log: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    ActionProbModel,
    Dict[str, torch.Tensor],
]:
    marginal_model_learner = KernelDensityRegressor(
        model=marginal_model,
        optimizer_kwargs={"lr": marginal_model_lr},
        device=device,
        random_seed=random_seed,
    )
    trained_logging_marginal_model, logging_marginal_model_training_logs = (
        marginal_model_learner.train_model_offline(
            dataset=logged_dataset,
            early_stage_policy=logging_early_stage_policy,
            late_stage_policy=logging_late_stage_policy,
            is_deterministic_early_stage=is_deterministic_early_stage_logging,
            is_deterministic_late_stage=is_deterministic_late_stage_logging,
            n_candidate_action=n_candidate_action_logging,
            kernel_function=gaussian_kernel,
            kernel_function_kwargs={
                "kernel_bandwidth": kernel_bandwidth,
            },
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            random_seed=random_seed,
        )
    )
    return trained_logging_marginal_model, logging_marginal_model_training_logs


def train_online_pg_policy(
    env: SyntheticDataGenerator,
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    credit_assignment_type: str,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Dict[str, torch.Tensor],
]:
    _, late_stage_policy_optimal = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )

    online_pg_learner = OnlinePolicyLearner(
        early_stage_policy=early_stage_policy,
        target_late_stage_policy=late_stage_policy_optimal,
        eval_late_stage_policy=late_stage_policy_optimal,
        early_stage_optimizer_kwargs={"lr": early_stage_lr},
        env=env,
        device=device,
        random_seed=random_seed,
    )
    trained_online_pg_early_stage_policy, online_pg_training_logs = (
        online_pg_learner.train_early_stage_policy_online(
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            credit_assignment_type=credit_assignment_type,  #
            is_deterministic_early_stage_eval=True,
            is_deterministic_late_stage_eval=True,
            n_candidate_action_train=n_candidate_action_train,  #
            n_candidate_action_eval=n_candidate_action_eval,  #
            random_seed=random_seed,
        )
    )
    return (
        trained_online_pg_early_stage_policy,
        online_pg_training_logs,
    )


def train_is_pg_policy(
    env: SyntheticDataGenerator,
    logged_dataset: LoggedDataset,
    logging_action_prob_model: ActionProbModel,
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    credit_assignment_type: str,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Dict[str, torch.Tensor],
]:
    _, late_stage_policy_optimal = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )

    is_pg_learner = ImportanceSamplingLearner(
        early_stage_policy=early_stage_policy,
        target_late_stage_policy=late_stage_policy_optimal,
        eval_late_stage_policy=late_stage_policy_optimal,
        is_model_free_target_late_stage_policy=True,
        is_model_free_eval_late_stage_policy=True,
        early_stage_optimizer_kwargs={"lr": early_stage_lr},
        env=env,
        device=device,
        random_seed=random_seed,
    )
    trained_is_pg_early_stage_policy, is_pg_training_logs = (
        is_pg_learner.train_early_stage_policy_offline(
            dataset=logged_dataset,
            logging_action_prob_model=logging_action_prob_model,
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            credit_assignment_type=credit_assignment_type,  #
            is_deterministic_early_stage_eval=True,
            is_deterministic_late_stage_eval=True,
            n_candidate_action_train=n_candidate_action_train,  #
            n_candidate_action_eval=n_candidate_action_eval,  #
            random_seed=random_seed,
        )
    )
    return (
        trained_is_pg_early_stage_policy,
        is_pg_training_logs,
    )


def train_kernel_is_pg_policy(
    env: SyntheticDataGenerator,
    logged_dataset: LoggedDataset,
    logging_marginal_model: KernelDensityModel,
    kernel_bandwidth: Union[int, float],
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    credit_assignment_type: str,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Dict[str, torch.Tensor],
]:
    _, late_stage_policy_optimal = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )

    kernel_is_pg_learner = KernelImportanceSamplingLearner(
        early_stage_policy=early_stage_policy,
        target_late_stage_policy=late_stage_policy_optimal,
        eval_late_stage_policy=late_stage_policy_optimal,
        early_stage_optimizer_kwargs={"lr": early_stage_lr},
        env=env,
        device=device,
        random_seed=random_seed,
    )
    trained_kernel_is_pg_early_stage_policy, kernel_is_pg_training_logs = (
        kernel_is_pg_learner.train_early_stage_policy_offline(
            dataset=logged_dataset,
            logging_marginal_model=logging_marginal_model,
            kernel_function=gaussian_kernel,
            kernel_function_kwargs={
                "kernel_bandwidth": kernel_bandwidth,
            },
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            credit_assignment_type=credit_assignment_type,  #
            is_deterministic_early_stage_eval=True,
            is_deterministic_late_stage_eval=True,
            n_candidate_action_train=n_candidate_action_train,  #
            n_candidate_action_eval=n_candidate_action_eval,  #
            random_seed=random_seed,
        )
    )
    return (
        trained_kernel_is_pg_early_stage_policy,
        kernel_is_pg_training_logs,
    )


def save_logs(
    rootdir: str,
    experiment_name: str,
    random_seed: int,
    logging_type: str,
    credit_assignment_type: Optional[str],
    n_candidate_action_train: Optional[int],
    setting: str,
    key_param: Optional[Union[str, int, float]],  # None in the default setting
    trained_logging_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    trained_logging_late_stage_policy: Optional[BaseLateStagePolicy] = None,
    trained_naive_cf_early_stage_policy: Optional[BaselineEarlyStagePolicy] = None,
    trained_naive_cf_late_stage_policy: Optional[BaselineLateStagePolicy] = None,
    trained_moe_cf_early_stage_policy: Optional[BaselineEarlyStagePolicy] = None,
    trained_moe_cf_model_selector: Optional[ExpertSelectionModel] = None,
    trained_quantile_cf_model: Optional[EarlyStageTwoTowerQuantileModel] = None,
    trained_online_pg_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    trained_is_pg_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    trained_kernel_is_pg_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    trained_logging_action_prob_model: Optional[ActionProbModel] = None,
    trained_logging_marginal_model: Optional[KernelDensityModel] = None,
    logging_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    naive_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    moe_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    quantile_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    late_stage_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    online_pg_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    is_pg_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    kernel_is_pg_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    logging_action_prob_model_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    logging_marginal_model_training_logs: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if experiment_name == "auto":
        experiment_name = setting

    # for logging policies path
    if key_param is not None:
        logging_rootdir = Path(f"{rootdir}/{experiment_name},param={key_param}")

    else:
        logging_rootdir = Path(f"{rootdir}/{experiment_name}")

    logging_rootdir.mkdir(parents=True, exist_ok=True)

    if (
        trained_logging_early_stage_policy is not None
        or trained_logging_late_stage_policy is not None
        or logging_training_logs is not None
    ):
        logging_dir = Path(logging_rootdir / "logging")
        logging_dir.mkdir(parents=True, exist_ok=True)

        if trained_logging_early_stage_policy is not None:
            logging_early_stage_policy_path = logging_dir / "early_stage_policy.pt"

            torch.save(
                trained_logging_early_stage_policy.base_model.state_dict(),
                logging_early_stage_policy_path,
            )

        if trained_logging_late_stage_policy is not None:
            logging_late_stage_policy_path = logging_dir / "late_stage_policy.pt"

            torch.save(
                trained_logging_late_stage_policy.base_model.state_dict(),
                logging_late_stage_policy_path,
            )

        if logging_training_logs is not None:
            logging_training_process_dir = Path(logging_dir / "training_process")
            logging_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in logging_training_logs.keys():
                logging_training_logs_path = logging_training_process_dir / f"{key}.pt"
                torch.save(logging_training_logs[key], logging_training_logs_path)

    # update the credit-assignment information in the path
    if key_param is not None:
        online_pg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={'na'},credit_assignment={credit_assignment_type},seed={random_seed}"
        )

    else:
        online_pg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},logging={'na'},credit_assignment={credit_assignment_type},seed={random_seed}"
        )

    if (
        trained_online_pg_early_stage_policy is not None
        or online_pg_training_logs is not None
    ):
        online_pg_dir = Path(online_pg_rootdir / "online")
        online_pg_dir.mkdir(parents=True, exist_ok=True)

        if trained_online_pg_early_stage_policy is not None:
            online_pg_early_stage_policy_path = online_pg_dir / "early_stage_policy.pt"

            torch.save(
                trained_online_pg_early_stage_policy.base_model.state_dict(),
                online_pg_early_stage_policy_path,
            )

        if online_pg_training_logs is not None:
            online_pg_training_process_dir = Path(online_pg_dir / "training_process")
            online_pg_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in online_pg_training_logs.keys():
                online_pg_training_logs_path = (
                    online_pg_training_process_dir / f"{key}.pt"
                )
                torch.save(online_pg_training_logs[key], online_pg_training_logs_path)

    # update the logging information in the path
    if key_param is not None:
        reg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},seed={random_seed}"
        )

    else:
        reg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={random_seed}"
        )

    rootdir.mkdir(parents=True, exist_ok=True)

    if (
        trained_naive_cf_early_stage_policy is not None
        or trained_naive_cf_late_stage_policy is not None
        or naive_cf_training_logs is not None
        or late_stage_cf_training_logs is not None
    ):
        naive_cf_dir = Path(reg_rootdir / "naive_cf")
        naive_cf_dir.mkdir(parents=True, exist_ok=True)

        if trained_naive_cf_early_stage_policy is not None:
            naive_cf_early_stage_policy_path = naive_cf_dir / "early_stage_policy.pt"

            torch.save(
                trained_naive_cf_early_stage_policy.base_model.state_dict(),
                naive_cf_early_stage_policy_path,
            )

        if trained_naive_cf_late_stage_policy is not None:
            naive_cf_late_stage_policy_path = naive_cf_dir / "late_stage_policy.pt"

            torch.save(
                trained_naive_cf_late_stage_policy.base_model.state_dict(),
                naive_cf_late_stage_policy_path,
            )

        if naive_cf_training_logs is not None:
            naive_cf_training_process_dir = Path(naive_cf_dir / "training_process")
            naive_cf_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in naive_cf_training_logs.keys():
                naive_cf_training_logs_path = (
                    naive_cf_training_process_dir / f"{key}.pt"
                )
                torch.save(naive_cf_training_logs[key], naive_cf_training_logs_path)

        if late_stage_cf_training_logs is not None:
            late_stage_cf_training_process_dir = Path(
                naive_cf_dir / "late_stage_training_process"
            )
            late_stage_cf_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in late_stage_cf_training_logs.keys():
                late_stage_cf_training_logs_path = (
                    late_stage_cf_training_process_dir / f"{key}.pt"
                )
                torch.save(
                    late_stage_cf_training_logs[key],
                    late_stage_cf_training_logs_path,
                )

    if (
        trained_moe_cf_early_stage_policy is not None
        or trained_moe_cf_model_selector is not None
        or moe_cf_training_logs is not None
    ):
        moe_cf_dir = Path(reg_rootdir / "moe_cf")
        moe_cf_dir.mkdir(parents=True, exist_ok=True)

        if trained_moe_cf_early_stage_policy is not None:
            moe_cf_early_stage_policy_path = moe_cf_dir / "early_stage_policy.pt"

            torch.save(
                trained_moe_cf_early_stage_policy.base_model.state_dict(),
                moe_cf_early_stage_policy_path,
            )

        if trained_moe_cf_model_selector is not None:
            moe_cf_model_selector_path = moe_cf_dir / "model_selector.pt"

            torch.save(
                trained_moe_cf_model_selector.state_dict(),
                moe_cf_model_selector_path,
            )

        if moe_cf_training_logs is not None:
            moe_cf_training_process_dir = Path(moe_cf_dir / "training_process")
            moe_cf_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in moe_cf_training_logs.keys():
                moe_cf_training_logs_path = moe_cf_training_process_dir / f"{key}.pt"
                torch.save(moe_cf_training_logs[key], moe_cf_training_logs_path)

    if trained_quantile_cf_model is not None or quantile_cf_training_logs is not None:
        quantile_cf_dir = Path(reg_rootdir / "quantile_cf")
        quantile_cf_dir.mkdir(parents=True, exist_ok=True)

        if trained_quantile_cf_model is not None:
            quantile_cf_model_path = quantile_cf_dir / "quantile_cf_model.pt"

            torch.save(
                trained_quantile_cf_model.state_dict(),
                quantile_cf_model_path,
            )

        if quantile_cf_training_logs is not None:
            quantile_cf_training_process_dir = Path(
                quantile_cf_dir / "training_process"
            )
            quantile_cf_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in quantile_cf_training_logs.keys():
                quantile_cf_training_logs_path = (
                    quantile_cf_training_process_dir / f"{key}.pt"
                )
                torch.save(
                    quantile_cf_training_logs[key], quantile_cf_training_logs_path
                )

    if (
        trained_logging_action_prob_model is not None
        or logging_action_prob_model_training_logs is not None
    ):
        logging_action_prob_model_dir = Path(reg_rootdir / "logging_action_prob_model")
        logging_action_prob_model_dir.mkdir(parents=True, exist_ok=True)

        if trained_logging_action_prob_model is not None:
            logging_action_prob_model_path = (
                logging_action_prob_model_dir / "logging_action_prob_model.pt"
            )

            torch.save(
                trained_logging_action_prob_model.state_dict(),
                logging_action_prob_model_path,
            )

        if logging_action_prob_model_training_logs is not None:
            logging_action_prob_model_training_process_dir = Path(
                logging_action_prob_model_dir / "training_process"
            )
            logging_action_prob_model_training_process_dir.mkdir(
                parents=True, exist_ok=True
            )

            for key in logging_action_prob_model_training_logs.keys():
                logging_action_prob_model_training_logs_path = (
                    logging_action_prob_model_training_process_dir / f"{key}.pt"
                )
                torch.save(
                    logging_action_prob_model_training_logs[key],
                    logging_action_prob_model_training_logs_path,
                )

    if (
        trained_logging_marginal_model is not None
        or logging_marginal_model_training_logs is not None
    ):
        logging_marginal_model_dir = Path(reg_rootdir / "logging_marginal_model")
        logging_marginal_model_dir.mkdir(parents=True, exist_ok=True)

        if trained_logging_marginal_model is not None:
            logging_marginal_model_path = (
                logging_marginal_model_dir / "logging_marginal_model.pt"
            )

            torch.save(
                trained_logging_marginal_model.state_dict(),
                logging_marginal_model_path,
            )

        if logging_marginal_model_training_logs is not None:
            logging_marginal_model_training_process_dir = Path(
                logging_marginal_model_dir / "training_process"
            )
            logging_marginal_model_training_process_dir.mkdir(
                parents=True, exist_ok=True
            )

            for key in logging_marginal_model_training_logs.keys():
                logging_marginal_model_training_logs_path = (
                    logging_marginal_model_training_process_dir / f"{key}.pt"
                )
                torch.save(
                    logging_marginal_model_training_logs[key],
                    logging_marginal_model_training_logs_path,
                )

    # update the logging, credit-assignment information in the path
    if key_param is not None:
        offline_pg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={logging_type},credit_assignment={credit_assignment_type},seed={random_seed}"
        )

    else:
        offline_pg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},logging={logging_type},credit_assignment={credit_assignment_type},seed={random_seed}"
        )

    if trained_is_pg_early_stage_policy is not None or is_pg_training_logs is not None:
        is_pg_dir = Path(offline_pg_rootdir / "IS")
        is_pg_dir.mkdir(parents=True, exist_ok=True)

        if trained_is_pg_early_stage_policy is not None:
            is_pg_early_stage_policy_path = is_pg_dir / "early_stage_policy.pt"

            torch.save(
                trained_is_pg_early_stage_policy.base_model.state_dict(),
                is_pg_early_stage_policy_path,
            )

        if is_pg_training_logs is not None:
            is_pg_training_process_dir = Path(is_pg_dir / "training_process")
            is_pg_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in is_pg_training_logs.keys():
                is_pg_training_logs_path = is_pg_training_process_dir / f"{key}.pt"
                torch.save(is_pg_training_logs[key], is_pg_training_logs_path)

    if (
        trained_kernel_is_pg_early_stage_policy is not None
        or kernel_is_pg_training_logs is not None
    ):
        kernel_is_pg_dir = Path(offline_pg_rootdir / "kernelIS")
        kernel_is_pg_dir.mkdir(parents=True, exist_ok=True)

        if trained_kernel_is_pg_early_stage_policy is not None:
            kernel_is_pg_early_stage_policy_path = (
                kernel_is_pg_dir / "early_stage_policy.pt"
            )

            torch.save(
                trained_kernel_is_pg_early_stage_policy.base_model.state_dict(),
                kernel_is_pg_early_stage_policy_path,
            )

        if kernel_is_pg_training_logs is not None:
            kernel_is_pg_training_process_dir = Path(
                kernel_is_pg_dir / "training_process"
            )
            kernel_is_pg_training_process_dir.mkdir(parents=True, exist_ok=True)

            for key in kernel_is_pg_training_logs.keys():
                kernel_is_pg_training_logs_path = (
                    kernel_is_pg_training_process_dir / f"{key}.pt"
                )
                torch.save(
                    kernel_is_pg_training_logs[key], kernel_is_pg_training_logs_path
                )


def initialize_trainable_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    n_moe_model: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[BaselineEarlyStagePolicy, BaselineLateStagePolicy]:
    early_stage_base_model = EarlyStageTwoTowerModel(
        n_context=env.context_sampler.n_discrete_context,
        n_action=env.action_set.n_action,
        dim_emb=dim_model_emb,
        n_model=n_moe_model,
    )
    trainable_early_stage_policy = BaselineEarlyStagePolicy(
        base_model=early_stage_base_model,
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    late_stage_base_model = LateStageNeuralModel(
        n_context=env.context_sampler.n_discrete_context,
        n_latent=env.latent_sampler.n_discrete_latent,
        n_action=env.action_set.n_action,
        dim_emb=dim_model_emb,
    )
    trainable_late_stage_policy = BaselineLateStagePolicy(
        base_model=late_stage_base_model,
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return trainable_early_stage_policy, trainable_late_stage_policy


def initialize_trainable_qunatile_model(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
):
    quantile_model = EarlyStageTwoTowerQuantileModel(
        n_context=env.context_sampler.n_discrete_context,
        n_action=env.action_set.n_action,
        dim_emb=dim_model_emb,
    )
    return quantile_model


def initialize_trainable_action_prob_model(
    env: SyntheticDataGenerator,
    base_model: BaselineLateStagePolicy,
):
    action_prob_model = ActionProbModel(
        base_context_encoder=base_model.context_encoder,
        base_latent_encoder=base_model.latent_encoder,
        base_action_encoder=base_model.action_encoder,
        n_output_action=env.n_output_action,
    )
    return action_prob_model


def initialize_trainable_marginal_model(
    env: SyntheticDataGenerator,
    base_model: BaselineLateStagePolicy,
):
    marginal_model = KernelDensityModel(
        base_context_encoder=base_model.context_encoder,
        base_latent_encoder=base_model.latent_encoder,
        base_action_encoder=base_model.action_encoder,
        n_output_action=env.n_output_action,
    )
    return marginal_model


def initialize_optimal_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[OptimalEarlyStagePolicy, OptimalLateStagePolicy]:
    early_stage_policy_optimal = OptimalEarlyStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    late_stage_policy_optimal = OptimalLateStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return early_stage_policy_optimal, late_stage_policy_optimal


def initialize_uniform_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[UniformEarlyStagePolicy, UniformLateStagePolicy]:
    uniform_early_stage_policy = UniformEarlyStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    uniform_late_stage_policy = UniformLateStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return uniform_early_stage_policy, uniform_late_stage_policy


def load_logging_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_logging_path: str,
    late_stage_logging_path: str,
    device: torch.device,
    random_seed: int,
) -> Tuple[BaselineEarlyStagePolicy, BaselineLateStagePolicy]:
    logging_early_stage_policy, logging_late_stage_policy = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    logging_early_stage_policy.base_model.load_state_dict(
        torch.load(early_stage_logging_path, map_location=device)
    )
    logging_late_stage_policy.base_model.load_state_dict(
        torch.load(late_stage_logging_path, map_location=device)
    )
    return logging_early_stage_policy, logging_late_stage_policy


def load_naive_cf_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_naive_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineEarlyStagePolicy:
    early_stage_naive_cf_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    early_stage_naive_cf_policy.base_model.load_state_dict(
        torch.load(early_stage_naive_cf_path, map_location=device)
    )
    return early_stage_naive_cf_policy


def load_pg_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_model_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineEarlyStagePolicy:
    early_stage_pg_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    early_stage_pg_policy.base_model.load_state_dict(
        torch.load(early_stage_model_path, map_location=device)
    )
    return early_stage_pg_policy


def load_late_stage_cf_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    late_stage_naive_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineLateStagePolicy:
    _, late_stage_naive_cf_policy = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        device=device,
        random_seed=random_seed,
    )
    late_stage_naive_cf_policy.base_model.load_state_dict(
        torch.load(late_stage_naive_cf_path, map_location=device)
    )
    return late_stage_naive_cf_policy


def load_moe_cf_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    n_moe_model: int,
    early_stage_moe_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineEarlyStagePolicy:
    early_stage_moe_cf_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=n_moe_model,
        device=device,
        random_seed=random_seed,
    )
    early_stage_moe_cf_policy.base_model.load_state_dict(
        torch.load(early_stage_moe_cf_path, map_location=device)
    )
    return early_stage_moe_cf_policy


def load_moe_model_selector(
    env: SyntheticDataGenerator,
    n_moe_model: int,
    early_stage_moe_model_selector_path: str,
    device: torch.device,
) -> ExpertSelectionModel:
    model_selector = ExpertSelectionModel(
        n_latent=env.latent_sampler.n_discrete_latent,
        n_model=n_moe_model,
    )
    model_selector.load_state_dict(
        torch.load(early_stage_moe_model_selector_path, map_location=device)
    )
    return model_selector


def load_greedy_algorithm(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_quantile_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> GreedySubsetEarlyStagePolicy:
    quantile_cf_model = initialize_trainable_qunatile_model(
        env=env,
        dim_model_emb=dim_model_emb,
    )
    quantile_cf_model.load_state_dict(
        torch.load(early_stage_quantile_cf_path, map_location=device)
    )
    early_stage_greedy_policy = GreedySubsetEarlyStagePolicy(
        base_quantile_model=quantile_cf_model,
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return early_stage_greedy_policy


def load_logging_action_prob_model(
    env: SyntheticDataGenerator,
    base_model: BaselineLateStagePolicy,
    logging_action_prob_model_path: str,
    device: torch.device,
) -> BaselineEarlyStagePolicy:
    logging_action_prob_model = initialize_trainable_action_prob_model(
        env=env,
        base_model=base_model,
    )
    logging_action_prob_model.load_state_dict(
        torch.load(logging_action_prob_model_path, map_location=device)
    )
    return logging_action_prob_model


def load_logging_marginal_model(
    env: SyntheticDataGenerator,
    base_model: BaselineLateStagePolicy,
    logging_marginal_model_path: str,
    device: torch.device,
) -> BaselineEarlyStagePolicy:
    logging_marginal_model = initialize_trainable_marginal_model(
        env=env,
        base_model=base_model,
    )
    logging_marginal_model.load_state_dict(
        torch.load(logging_marginal_model_path, map_location=device)
    )
    return logging_marginal_model


def optimize_moe_model_assignment(
    logged_dataset: LoggedDataset,
    model_selector: ExpertSelectionModel,
    n_candidate_action_eval: int,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        latent_id = logged_dataset.latent_id.to(device)
        model_prob = model_selector(latent_id)

    n_model = model_prob.shape[-1]
    model_id = model_prob.max(dim=-1)[1]

    counter = Counter(model_id.tolist())
    model_prob = torch.tensor(
        [counter[i] / len(model_id) for i in range(n_model)], device=device
    )

    raw_counts = model_prob * n_candidate_action_eval
    assignments = torch.floor(raw_counts).to(int)

    remainders = raw_counts - assignments
    _, indices = torch.topk(remainders, n_candidate_action_eval - assignments.sum())
    assignments[indices] += 1

    return assignments
