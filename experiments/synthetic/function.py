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
    CollaborativeFilteringLearner,
    OnlinePolicyLearner,
)
from synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
    BaselineEarlyStagePolicy,
    BaselineLateStagePolicy,
    EarlyStageTwoTowerModel,
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
    Dict[str, torch.Tensor],
]:
    cf_learner = CollaborativeFilteringLearner(
        early_stage_policy=early_stage_policy,
        late_stage_policy=late_stage_policy,
        early_stage_optimizer_kwargs={
            "lr": early_stage_lr,
        },
        late_stage_optimizer_kwargs={
            "lr": late_stage_lr,
        },
        env=env,
        device=device,
        random_seed=random_seed,
    )
    (
        trained_early_stage_policy,
        trained_late_stage_policy,
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
        is_model_free_late_stage_policy=True,
        early_stage_optimizer_kwargs={
            "lr": early_stage_lr,
        },
        env=env,
        device=device,
        random_seed=random_seed,
    )
    (
        trained_early_stage_policy,
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


def save_logs(
    rootdir: str,
    experiment_name: str,
    random_seed: int,
    logging_type: str,
    credit_assignment_type: Optional[str],
    n_candidate_action_train: Optional[int],
    setting: str,
    key_param: Optional[Union[str, int, float]],  # None in the default setting
    trained_naive_cf_early_stage_policy: Optional[BaselineEarlyStagePolicy] = None,
    trained_naive_cf_late_stage_policy: Optional[BaselineLateStagePolicy] = None,
    trained_online_pg_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    naive_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    late_stage_cf_training_logs: Optional[Dict[str, torch.Tensor]] = None,
    online_pg_training_logs: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if experiment_name == "auto":
        experiment_name = setting

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
            f"{rootdir}/{experiment_name}/{experiment_name},param={key_param},logging={'uniform'},seed={random_seed}"
        )

    else:
        reg_rootdir = Path(
            f"{rootdir}/{experiment_name}/{experiment_name},logging={'uniform'},seed={random_seed}"
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
