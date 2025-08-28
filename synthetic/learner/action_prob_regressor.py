# pyre-unsafe
"""Learner class to regress the action choice probability of the joint policy."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from early_stage_retrieval.synthetic.dataset import (
    LoggedDataset,
)
from early_stage_retrieval.synthetic.policy.base import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
    BaseSingleStagePolicy,
)
from torch import nn
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .base import BaseModelLearner


@dataclass
class ActionProbRegressor(BaseModelLearner):
    """Training procedure of the action probability reegression model.

    Input
    ------
    model: nn.Module
        The model to train. Either model or joint_policy should be provided.

    optimizer: Optimizer = Adagrad
        The optimizer to use for the training.

    optimizer_kwargs: Optional[Dict[str, Any]]
        The optimizer kwargs to use for the training.

    device: torch.device, default=torch.device("cpu")
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    model: nn.Module
    optimizer: Optimizer = Adagrad  # pyre-ignore
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    device: torch.device = torch.device("cpu")
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.model = self.model.to(self.device)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {"lr": 1e-2}

    def train_model_offline(
        self,
        dataset: LoggedDataset,
        policy: Optional[BaseSingleStagePolicy] = None,
        early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
        late_stage_policy: Optional[BaseLateStagePolicy] = None,
        is_deterministic_single_stage: bool = False,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_candidate_action: int = 10,
        n_candidate_per_model: Optional[List[int]] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """Train the base model used in the early stage policy.

        Input
        ------
        dataset: LoggedDataset
            The training data, which contains the following keys:

            .. code-block:: python

                key: [
                    "context",     # (n_samples, dim_context)
                    "latent",      # (n_samples, dim_hidden, dim_action_emb)
                    "action",      # (n_samples, n_output_action)
                    "reward",      # (n_samples, n_output_action)
                    "agg_reward",  # (n_samples, 1)
                    "context_id",  # (n_samples, ), optional
                    "latent_id",   # (n_samples, ), optional
                    ]

        policy: BaseSingleStagePolicy
            The single stage policy. Either policy or (early stage policy, late stage policy) should be provided.

        early_stage_policy: BaseEarlyStagePolicy
            The early stage policy. Either policy or (early stage policy, late stage policy) should be provided.

        late_stage_policy: BaseLateStagePolicy
            The late stage policy. Either policy or (early stage policy, late stage policy) should be provided.

        is_deterministic_single_stage: bool, default=False
            Whether the single stage policy is deterministic.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage logging policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage logging policy is deterministic.

        n_candidate_action: int, default=10
            The number of candidate actions to sample from the early stage policy.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to sample per model.

        val_ratio: float, default=0.1
            The ratio of validation data.

        test_ratio: float, default=0.1
            The ratio of test data.

        n_epoch: int, default=1000
            The number of epochs to train.

        n_steps_per_epoch: int, default=10
            The number of steps per epoch.

        n_epochs_per_log: int, default=10
            The number of epochs per log.

        patience: int, default=5
            The number of epochs to wait before early stopping.

        batch_size: int, default=128
            The batch size.

        make_copy: bool, default=False
            Whether to make a copy of the base model.

        return_training_logs: bool, default=False
            Whether to return the training logs.

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        """
        if policy is None and (
            early_stage_policy is None
            or late_stage_policy is None
            or n_candidate_action is None
        ):
            raise ValueError(
                "Either policy or (early stage policy, late stage policy, n_candidate_action) should be provided."
            )

        if random_seed is None:
            random_seed = self.random_seed

        self.seed(random_seed)  # pyre-ignore

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(  # pyre-ignore
            model.parameters(),
            **self.optimizer_kwargs,
        )

        val_size = int(val_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        train_size = len(dataset) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

        train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        val_losses = torch.zeros((n_epoch // n_epochs_per_log + 1,), device=self.device)

        loss_fn = nn.MSELoss()

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[Train action prob regression model: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
                    },
                )

                train_iterator = iter(train_loader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch_ = next(train_iterator)

                    context_ = batch_["context"].to(self.device, non_blocking=True)
                    latent_ = batch_["latent"].to(self.device, non_blocking=True)
                    context_id_ = batch_["context_id"].to(
                        self.device, non_blocking=True
                    )
                    latent_id_ = batch_["latent_id"].to(self.device, non_blocking=True)
                    action_ids_ = batch_["action"].to(self.device, non_blocking=True)

                    with torch.no_grad():
                        if policy is not None:
                            actions_ = policy.sample(
                                context=context_,
                                context_id=context_id_,
                                latent=latent_,
                                latent_id=latent_id_,
                                is_deterministic=is_deterministic_single_stage,
                            )
                        else:
                            candidate_actions_ = (
                                early_stage_policy.sample(  # pyre-ignore
                                    context=context_,
                                    context_id=context_id_,
                                    n_candidate_action=n_candidate_action,
                                    n_candidate_per_model=n_candidate_per_model,
                                    is_deterministic=is_deterministic_early_stage,
                                )
                            )
                            actions_ = late_stage_policy.sample(  # pyre-ignore
                                candidate_actions=candidate_actions_,
                                context=context_,
                                context_id=context_id_,
                                latent=latent_,
                                latent_id=latent_id_,
                                is_deterministic=is_deterministic_late_stage,
                            )

                    labels_ = (actions_ == action_ids_).float()

                    preds_ = model(
                        context=context_,
                        context_id=context_id_,
                        latent=latent_,
                        latent_id=latent_id_,
                        actions=action_ids_,
                    )

                    loss_ = loss_fn(preds_, labels_)
                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_ / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    with torch.no_grad():
                        for batch_ in val_loader:
                            context_ = batch_["context"].to(
                                self.device, non_blocking=True
                            )
                            latent_ = batch_["latent"].to(
                                self.device, non_blocking=True
                            )
                            context_id_ = batch_["context_id"].to(
                                self.device, non_blocking=True
                            )
                            latent_id_ = batch_["latent_id"].to(
                                self.device, non_blocking=True
                            )
                            action_ids_ = batch_["action"].to(
                                self.device, non_blocking=True
                            )

                            if policy is not None:
                                actions_ = policy.sample(
                                    context=context_,
                                    context_id=context_id_,
                                    latent=latent_,
                                    latent_id=latent_id_,
                                    is_deterministic=is_deterministic_single_stage,
                                )
                            else:
                                candidate_actions_ = early_stage_policy.sample(
                                    context=context_,
                                    context_id=context_id_,
                                    n_candidate_action=n_candidate_action,
                                    n_candidate_per_model=n_candidate_per_model,
                                    is_deterministic=is_deterministic_early_stage,
                                )
                                actions_ = late_stage_policy.sample(
                                    candidate_actions=candidate_actions_,
                                    context=context_,
                                    context_id=context_id_,
                                    latent=latent_,
                                    latent_id=latent_id_,
                                    is_deterministic=is_deterministic_late_stage,
                                )

                            labels_ = (actions_ == action_ids_).float()

                            preds_ = model(
                                context=context_,
                                context_id=context_id_,
                                latent=latent_,
                                latent_id=latent_id_,
                                actions=action_ids_,
                            )
                            val_loss_ = loss_fn(preds_, labels_)

                            val_losses[(i + 1) // n_epochs_per_log] += (
                                val_loss_ * len(batch_["context_id"]) / len(val_dataset)
                            )

                    # early stopping
                    current_epoch = (i + 1) // n_epochs_per_log
                    best_loss_epoch = torch.argmin(
                        val_losses[1 : current_epoch + 1]
                    ).item()

                    if current_epoch - best_loss_epoch > patience:  # pyre-ignore
                        print("early stopping at epoch", i + 1)
                        break

        self.trained_model = model  # pyre-ignore

        if save_path is not None:
            self.save_model(save_path)

        if return_training_logs:
            output = (
                model,
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
            )
        else:
            output = model

        return output
