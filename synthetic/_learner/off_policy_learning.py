"""Traning procedure of off-policy learning."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from synthetic.dataset import (
    BaseDataGenerator,
    LoggedDataset,
)
from synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseJointPolicy,
    BaseLateStagePolicy,
    BaseSingleStagePolicy,
)
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .base import BasePolicyLearner
from .kernel_function import gaussian_kernel


@dataclass
class ImportanceSamplingLearner(BasePolicyLearner):
    def train_early_stage_policy_offline(self, **kwargs):
        raise NotImplementedError()


@dataclass
class KernelImportanceSamplingLearner(BasePolicyLearner):
    def train_early_stage_policy_offline(self, **kwargs):
        raise NotImplementedError()


# @dataclass
# class ImportanceSamplingLearner(BasePolicyLearner):
#     """Training procedure of off-policy learning (vanilla importance sampling).

#     Input
#     ------
#     early_stage_policy: BaseEarlyStagePolicy
#         The early stage policy.

#     target_late_stage_policy: BaseLateStagePolicy
#         The late stage policy used for training.

#     eval_late_stage_policy: BaseLateStagePolicy
#         The late stage policy used for evaluation.

#     is_model_free_target_late_stage_policy: bool, default=False
#         Whether the target late stage policy is model-free.

#     is_model_free_eval_late_stage_policy: bool, default=False
#         Whether the evaluation late stage policy is model-free.

#     env: Optional[BaseDataGenerator]
#         The data generation environment.

#     early_stage_optimizer: Optimizer = Adagrad
#         The optimizer to use for the early stage policy.

#     early_stage_optimizer_kwargs: Optional[Dict[str, Any]]
#         The optimizer kwargs to use for the early stage policy.

#     device: torch.device, default=torch.device("cpu")
#         The device to use.

#     random_seed: Optional[int]
#         The random seed to use.

#     """

#     early_stage_policy: BaseEarlyStagePolicy
#     target_late_stage_policy: BaseLateStagePolicy
#     eval_late_stage_policy: BaseLateStagePolicy
#     is_model_free_target_late_stage_policy: bool = False
#     is_model_free_eval_late_stage_policy: bool = False
#     env: Optional[BaseDataGenerator] = None
#     early_stage_optimizer: Optimizer = Adagrad  # pyre-ignore
#     early_stage_optimizer_kwargs: Optional[Dict[str, Any]] = None
#     device: torch.device = torch.device("cpu")
#     random_seed: Optional[int] = None

#     def __post_init__(self):
#         torch.manual_seed(self.random_seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(self.random_seed)

#             self.early_stage_policy.base_model = self.early_stage_policy.base_model.to(
#                 self.device
#             )

#             if not self.is_model_free_target_late_stage_policy:
#                 self.target_late_stage_policy.base_model = (
#                     self.target_late_stage_policy.base_model.to(self.device)
#                 )
#             if not self.is_model_free_eval_late_stage_policy:
#                 self.eval_late_stage_policy.base_model = (
#                     self.eval_late_stage_policy.base_model.to(self.device)
#                 )

#         if self.early_stage_optimizer_kwargs is None:
#             self.early_stage_optimizer_kwargs = {"lr": 1e-2}

#     def _policy_gradient(
#         self,
#         early_stage_log_prob: torch.Tensor,
#         late_stage_prob: torch.Tensor,
#         logging_prob: torch.Tensor,
#         agg_reward: torch.Tensor,
#     ):
#         """Compute the policy gradient.

#         Input
#         ------
#         early_stage_log_prob: torch.Tensor, shape (n_samples, )
#             The log probability of the candidate action (differential).

#         late_stage_prob: torch.Tensor, shape (n_samples, )
#             The probability of the action given candidate (non-differential).

#         logging_prob: torch.Tensor, shape (n_samples, )
#             The log probability of the action (non-differential).

#         agg_reward: torch.Tensor, shape (n_samples, )
#             The aggregated reward (non-differential).

#         Output
#         ------
#         policy_gradient: torch.Tensor, shape (1, )
#             The policy gradient.

#         """
#         iw = late_stage_prob / logging_prob
#         policy_gradient = -iw * early_stage_log_prob * agg_reward
#         return policy_gradient.mean()

#     def train_early_stage_policy_offline(  # pyre-ignore
#         self,
#         dataset: LoggedDataset,
#         logging_policy: Union[BaseJointPolicy, BaseSingleStagePolicy],
#         n_candidate_action: int = 10,
#         n_candidate_per_model: Optional[List[int]] = None,
#         val_ratio: float = 0.1,
#         test_ratio: float = 0.1,
#         n_epoch: int = 1000,
#         n_steps_per_epoch: int = 10,
#         n_epochs_per_log: int = 10,
#         patience: int = 5,
#         batch_size: int = 128,
#         make_copy: bool = False,
#         return_training_logs: bool = False,
#         is_deterministic_early_stage_eval: bool = False,
#         is_deterministic_late_stage_eval: bool = False,
#         save_path: Optional[Path] = None,
#         random_seed: Optional[int] = None,
#     ):
#         """Train the base model used in the early stage policy.

#         Input
#         ------
#         dataset: LoggedDataset
#             The training data, which contains the following keys:

#             .. code-block:: python

#                 key: [
#                     "context",     # (n_samples, dim_context)
#                     "latent",      # (n_samples, dim_hidden, dim_action_emb)
#                     "action",      # (n_samples, n_output_action)
#                     "reward",      # (n_samples, n_output_action)
#                     "agg_reward",  # (n_samples, 1)
#                     "context_id",  # (n_samples, ), optional
#                     "latent_id",   # (n_samples, ), optional
#                     ]

#         logging_policy: BaseJointPolicy
#             The logging policy.

#         n_candidate_action: int, default=10
#             The number of candidate actions to sample.

#         val_ratio: float, default=0.1
#             The ratio of validation data.

#         test_ratio: float, default=0.1
#             The ratio of test data.

#         n_epoch: int, default=1000
#             The number of epochs to train.

#         n_steps_per_epoch: int, default=10
#             The number of steps per epoch.

#         n_epochs_per_log: int, default=10
#             The number of epochs per log.

#         patience: int, default=5
#             The number of epochs to wait before early stopping.

#         batch_size: int, default=128
#             The batch size.

#         make_copy: bool, default=False
#             Whether to make a copy of the base model.

#         return_training_logs: bool, default=False
#             Whether to return the training logs.

#         is_deterministic_early_stage_eval: bool, default=False
#             Whether to use deterministic policy for evaluation.

#         is_deterministic_late_stage_eval: bool, default=False
#             Whether to use deterministic policy for evaluation.

#         save_path: Optional[Path]
#             The path to save the model.

#         random_seed: Optional[int]
#             The random seed to use.

#         """
#         if random_seed is None:
#             random_seed = self.random_seed

#         self.seed(random_seed)  # pyre-ignore

#         if make_copy:
#             early_stage_policy = deepcopy(self.early_stage_policy)
#         else:
#             early_stage_policy = self.early_stage_policy

#         optimizer = self.early_stage_optimizer(  # pyre-ignore
#             early_stage_policy.base_model.parameters(),  # pyre-ignore
#             **self.early_stage_optimizer_kwargs,
#         )

#         val_size = int(val_ratio * len(dataset))
#         test_size = int(test_ratio * len(dataset))
#         train_size = len(dataset) - val_size - test_size

#         train_dataset, val_dataset, test_dataset = random_split(
#             dataset, [train_size, val_size, test_size]
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
#         )
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

#         train_losses = torch.zeros((n_epoch + 1,), device=self.device)
#         val_losses = torch.zeros((n_epoch // n_epochs_per_log + 1,), device=self.device)
#         policy_values = torch.zeros(
#             (n_epoch // n_epochs_per_log + 1,), device=self.device
#         )

#         with tqdm(range(n_epoch)) as pbar:
#             for i, ch in enumerate(pbar):
#                 pbar.set_description(f"[Train base early stage model: Epoch {i}]")
#                 pbar.set_postfix(
#                     {
#                         "train_loss": f"{train_losses[i]:.4g}",
#                         "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
#                         "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
#                     },
#                 )

#                 train_iterator = iter(train_loader)

#                 for j in range(n_steps_per_epoch):
#                     try:
#                         batch_ = next(train_iterator)
#                     except StopIteration:
#                         train_iterator = iter(train_loader)
#                         batch_ = next(train_iterator)

#                     context_ = batch_["context"].to(self.device, non_blocking=True)
#                     latent_ = batch_["latent"].to(self.device, non_blocking=True)
#                     context_id_ = batch_["context_id"].to(
#                         self.device, non_blocking=True
#                     )
#                     latent_id_ = batch_["latent_id"].to(self.device, non_blocking=True)
#                     action_ids_ = batch_["action"].to(self.device, non_blocking=True)
#                     agg_reward_ = batch_["agg_reward"].to(
#                         self.device, non_blocking=True
#                     )
#                     factual_rewards_ = batch_["factual_rewards"].to(
#                         self.device, non_blocking=True
#                     )

#                     with torch.no_grad():
#                         logging_prob_, _ = logging_policy.calc_prob_given_actions(
#                             context=context_,
#                             context_id=context_id_,
#                             latent=latent_,
#                             latent_id=latent_id_,
#                             actions=action_ids_,
#                         )

#                     candidate_actions_, _, early_stage_log_prob_ = (
#                         early_stage_policy.sample_actions_with_prob(
#                             context=context_,
#                             context_id=context_id_,
#                             n_candidate_action=n_candidate_action,
#                             n_candidate_per_model=n_candidate_per_model,
#                         )
#                     )

#                     with torch.no_grad():
#                         late_stage_prob_, _ = (
#                             self.target_late_stage_policy.calc_prob_given_actions(
#                                 context=context_,
#                                 context_id=context_id_,
#                                 latent=latent_,
#                                 latent_id=latent_id_,
#                                 actions=action_ids_,
#                                 candidate_actions=candidate_actions_,
#                                 factual_rewards=factual_rewards_,
#                             )
#                         )

#                     loss_ = self._policy_gradient(
#                         early_stage_log_prob=early_stage_log_prob_,
#                         late_stage_prob=late_stage_prob_,
#                         logging_prob=logging_prob_,
#                         agg_reward=agg_reward_,
#                     )

#                     optimizer.zero_grad()

#                     with torch.autograd.set_detect_anomaly(True):
#                         loss_.backward()

#                     optimizer.step()

#                     train_losses[i + 1] += loss_ / n_steps_per_epoch

#                 if (i + 1) % n_epochs_per_log == 0:
#                     with torch.no_grad():
#                         for batch_ in val_loader:
#                             context_ = batch_["context"].to(
#                                 self.device, non_blocking=True
#                             )
#                             latent_ = batch_["latent"].to(
#                                 self.device, non_blocking=True
#                             )
#                             context_id_ = batch_["context_id"].to(
#                                 self.device, non_blocking=True
#                             )
#                             latent_id_ = batch_["latent_id"].to(
#                                 self.device, non_blocking=True
#                             )
#                             action_ids_ = batch_["action"].to(
#                                 self.device, non_blocking=True
#                             )
#                             agg_reward_ = batch_["agg_reward"].to(
#                                 self.device, non_blocking=True
#                             )
#                             factual_rewards_ = batch_["factual_rewards"].to(
#                                 self.device, non_blocking=True
#                             )

#                             logging_prob_, _ = logging_policy.calc_prob_given_actions(
#                                 context=context_,
#                                 context_id=context_id_,
#                                 latent=latent_,
#                                 latent_id=latent_id_,
#                                 actions=action_ids_,
#                             )
#                             candidate_actions_, _, early_stage_log_prob_ = (
#                                 early_stage_policy.sample_actions_with_prob(
#                                     context=context_,
#                                     context_id=context_id_,
#                                     n_candidate_action=n_candidate_action,
#                                     n_candidate_per_model=n_candidate_per_model,
#                                 )
#                             )
#                             late_stage_prob_, _ = (
#                                 self.target_late_stage_policy.calc_prob_given_actions(
#                                     context=context_,
#                                     context_id=context_id_,
#                                     latent=latent_,
#                                     latent_id=latent_id_,
#                                     actions=action_ids_,
#                                     candidate_actions=candidate_actions_,
#                                     factual_rewards=factual_rewards_,
#                                 )
#                             )

#                             val_loss_ = self._policy_gradient(
#                                 early_stage_log_prob=early_stage_log_prob_,
#                                 late_stage_prob=late_stage_prob_,
#                                 logging_prob=logging_prob_,
#                                 agg_reward=agg_reward_,
#                             )

#                             val_losses[(i + 1) // n_epochs_per_log] += (
#                                 val_loss_ * len(batch_["context_id"]) / len(val_dataset)
#                             )

#                     if self.env is not None:
#                         policy_value_ = self.env.evaluate_policy_online(  # pyre-ignore
#                             early_stage_policy=early_stage_policy,
#                             late_stage_policy=self.eval_late_stage_policy,
#                             is_deterministic_early_stage=is_deterministic_early_stage_eval,
#                             is_deterministic_late_stage=is_deterministic_late_stage_eval,
#                             n_candidate_action=n_candidate_action,
#                             n_candidate_per_model=n_candidate_per_model,
#                         )
#                         policy_values[(i + 1) // n_epochs_per_log] += (
#                             policy_value_.item()
#                         )

#                     # early stopping
#                     current_epoch = (i + 1) // n_epochs_per_log
#                     best_loss_epoch = torch.argmin(
#                         val_losses[1 : current_epoch + 1]
#                     ).item()

#                     if current_epoch - best_loss_epoch > patience:  # pyre-ignore
#                         print("early stopping at epoch", i + 1)
#                         break

#         self.trained_early_stage_policy = early_stage_policy  # pyre-ignore

#         if save_path is not None:
#             self.save_early_stage_model(save_path)

#         if return_training_logs:
#             output = (
#                 early_stage_policy,
#                 {
#                     "train_losses": train_losses,
#                     "val_losses": val_losses,
#                     "policy_values": policy_values,
#                 },
#             )
#         else:
#             output = early_stage_policy

#         return output


# @dataclass
# class KernelImportanceSamplingLearner(BasePolicyLearner):
#     """Training procedure of off-policy learning (kernell importance sampling).

#     Input
#     ------
#     early_stage_policy: BaseEarlyStagePolicy
#         The early stage policy.

#     target_late_stage_policy: BaseLateStagePolicy
#         The late stage policy.

#     eval_late_stage_policy: BaseLateStagePolicy
#         The late stage policy for evaluation.

#     action_embedding_model: nn.Module
#         The (pre-trained) action embedding model.

#     distance_model: nn.Module
#         The (pre-trained) distance model.

#     is_model_free_target_late_stage_policy: bool, default=False
#         Whether the target late stage policy is model-free.

#     is_model_free_eval_late_stage_policy: bool, default=False
#         Whether the evaluation late stage policy is model-free.

#     use_raw_action_embedding: bool, default=False
#         Whether to use raw action embedding.

#     env: Optional[BaseDataGenerator]
#         The data generation environment.

#     early_stage_optimizer: Optimizer = Adagrad
#         The optimizer to use for the early stage policy.

#     early_stage_optimizer_kwargs: Optional[Dict[str, Any]]
#         The optimizer kwargs to use for the early stage policy.

#     device: torch.device, default=torch.device("cpu")
#         The device to use.

#     random_seed: Optional[int]
#         The random seed to use.

#     """

#     early_stage_policy: BaseEarlyStagePolicy
#     target_late_stage_policy: BaseLateStagePolicy
#     eval_late_stage_policy: BaseLateStagePolicy
#     action_embedding_model: nn.Module
#     distance_model: nn.Module
#     is_model_free_target_late_stage_policy: bool = False
#     is_model_free_eval_late_stage_policy: bool = False
#     use_raw_action_embedding: bool = False
#     env: Optional[BaseDataGenerator] = None
#     early_stage_optimizer: Optimizer = Adagrad  # pyre-ignore
#     early_stage_optimizer_kwargs: Optional[Dict[str, Any]] = None
#     device: torch.device = torch.device("cpu")
#     random_seed: Optional[int] = None

#     def __post_init__(self):
#         torch.manual_seed(self.random_seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(self.random_seed)

#             self.early_stage_policy.base_model = self.early_stage_policy.base_model.to(
#                 self.device
#             )

#             if not self.is_model_free_target_late_stage_policy:
#                 self.target_late_stage_policy.base_model = (
#                     self.target_late_stage_policy.base_model.to(self.device)
#                 )
#             if not self.is_model_free_eval_late_stage_policy:
#                 self.eval_late_stage_policy.base_model = (
#                     self.eval_late_stage_policy.base_model.to(self.device)
#                 )

#             self.action_embedding_model = self.action_embedding_model.to(self.device)
#             self.distance_model = self.distance_model.to(self.device)

#         if self.early_stage_optimizer_kwargs is None:
#             self.early_stage_optimizer_kwargs = {"lr": 1e-2}

#     def _policy_gradient(
#         self,
#         early_stage_log_prob: torch.Tensor,
#         kernel_weight: torch.Tensor,
#         logging_marginal_prob: torch.Tensor,
#         agg_reward: torch.Tensor,
#     ):
#         """Compute the policy gradient.

#         Input
#         ------
#         early_stage_log_prob: torch.Tensor, shape (n_samples, )
#             The log probability of the candidate action (differential).

#         kernel_weight: torch.Tensor, shape (n_samples, )
#             The kernel similarity weight between the logging action and the augmented one (non-differential).

#         logging_marginal_prob: torch.Tensor, shape (n_samples, )
#             The log marginal probability of the action (non-differential).

#         agg_reward: torch.Tensor, shape (n_samples, )
#             The aggregated reward (non-differential).

#         Output
#         ------
#         policy_gradient: torch.Tensor, shape (1, )
#             The policy gradient.

#         """
#         iw = kernel_weight / logging_marginal_prob
#         policy_gradient = -iw * early_stage_log_prob * agg_reward
#         return policy_gradient.mean()

#     def train_early_stage_policy_offline(  # pyre-ignore
#         self,
#         dataset: LoggedDataset,
#         logging_marginal_model: nn.Module,
#         kernel_function: Callable = gaussian_kernel,
#         kernel_function_kwargs: Optional[Dict[str, Any]] = None,
#         n_candidate_action: int = 10,
#         n_candidate_per_model: Optional[List[int]] = None,
#         val_ratio: float = 0.1,
#         test_ratio: float = 0.1,
#         n_epoch: int = 1000,
#         n_steps_per_epoch: int = 10,
#         n_epochs_per_log: int = 10,
#         patience: int = 5,
#         batch_size: int = 128,
#         make_copy: bool = False,
#         return_training_logs: bool = False,
#         is_deterministic_early_stage_eval: bool = False,
#         is_deterministic_late_stage_eval: bool = False,
#         save_path: Optional[Path] = None,
#         random_seed: Optional[int] = None,
#     ):
#         """Train the base model used in the early stage policy.

#         Input
#         ------
#         dataset: LoggedDataset
#             The training data, which contains the following keys:

#             .. code-block:: python

#                 key: [
#                     "context",     # (n_samples, dim_context)
#                     "latent",      # (n_samples, dim_hidden, dim_action_emb)
#                     "action",      # (n_samples, n_output_action)
#                     "reward",      # (n_samples, n_output_action)
#                     "agg_reward",  # (n_samples, 1)
#                     "context_id",  # (n_samples, ), optional
#                     "latent_id",   # (n_samples, ), optional
#                     ]

#         logging_marginal_model: nn.Module
#             The regression model to estimate the logging marginal density.

#         kernel_function: Callable, default=gaussian_kernel
#             Kernel function to use for the kernel importance sampling.

#         kernel_function_kwargs: Dict[str, Any], default=None
#             The kwargs to use for the kernel function.

#         n_candidate_action: int, default=10
#             The number of candidate actions to sample.

#         n_candidate_per_model: List[int], default=None
#             The number of candidate actions to sample per model.

#         val_ratio: float, default=0.1
#             The ratio of validation data.

#         test_ratio: float, default=0.1
#             The ratio of test data.

#         n_epoch: int, default=1000
#             The number of epochs to train.

#         n_steps_per_epoch: int, default=10
#             The number of steps per epoch.

#         n_epochs_per_log: int, default=10
#             The number of epochs per log.

#         patience: int, default=5
#             The number of epochs to wait before early stopping.

#         batch_size: int, default=128
#             The batch size.

#         make_copy: bool, default=False
#             Whether to make a copy of the base model.

#         return_training_logs: bool, default=False
#             Whether to return the training logs.

#         is_deterministic_early_stage_eval: bool, default=False
#             Whether to use deterministic policy for evaluation.

#         is_deterministic_late_stage_eval: bool, default=False
#             Whether to use deterministic policy for evaluation.di

#         save_path: Optional[Path]
#             The path to save the model.

#         random_seed: Optional[int]
#             The random seed to use.

#         """
#         if random_seed is None:
#             random_seed = self.random_seed

#         self.seed(random_seed)  # pyre-ignore

#         if make_copy:
#             early_stage_policy = deepcopy(self.early_stage_policy)
#         else:
#             early_stage_policy = self.early_stage_policy

#         optimizer = self.early_stage_optimizer(  # pyre-ignore
#             early_stage_policy.base_model.parameters(),  # pyre-ignore
#             **self.early_stage_optimizer_kwargs,
#         )

#         val_size = int(val_ratio * len(dataset))
#         test_size = int(test_ratio * len(dataset))
#         train_size = len(dataset) - val_size - test_size

#         train_dataset, val_dataset, test_dataset = random_split(
#             dataset, [train_size, val_size, test_size]
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
#         )
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

#         train_losses = torch.zeros((n_epoch + 1,), device=self.device)
#         val_losses = torch.zeros((n_epoch // n_epochs_per_log + 1,), device=self.device)
#         policy_values = torch.zeros(
#             (n_epoch // n_epochs_per_log + 1,), device=self.device
#         )

#         with tqdm(range(n_epoch)) as pbar:
#             for i, ch in enumerate(pbar):
#                 pbar.set_description(f"[Train base early stage model: Epoch {i}]")
#                 pbar.set_postfix(
#                     {
#                         "train_loss": f"{train_losses[i]:.4g}",
#                         "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
#                         "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
#                     },
#                 )

#                 train_iterator = iter(train_loader)

#                 for j in range(n_steps_per_epoch):
#                     try:
#                         batch_ = next(train_iterator)
#                     except StopIteration:
#                         train_iterator = iter(train_loader)
#                         batch_ = next(train_iterator)

#                     context_ = batch_["context"].to(self.device, non_blocking=True)
#                     latent_ = batch_["latent"].to(self.device, non_blocking=True)
#                     context_id_ = batch_["context_id"].to(
#                         self.device, non_blocking=True
#                     )
#                     latent_id_ = batch_["latent_id"].to(self.device, non_blocking=True)
#                     action_ids_ = batch_["action"].to(self.device, non_blocking=True)
#                     agg_reward_ = batch_["agg_reward"].to(
#                         self.device, non_blocking=True
#                     )
#                     factual_rewards_ = batch_["factual_rewards"].to(
#                         self.device, non_blocking=True
#                     )

#                     with torch.no_grad():
#                         logging_marginal_prob_ = logging_marginal_model(
#                             context=context_,
#                             context_id=context_id_,
#                             latent=latent_,
#                             latent_id=latent_id_,
#                             actions=action_ids_,
#                         )

#                     candidate_actions_, _, early_stage_log_prob_ = (
#                         early_stage_policy.sample_actions_with_prob(
#                             context=context_,
#                             context_id=context_id_,
#                             n_candidate_action=n_candidate_action,
#                             n_candidate_per_model=n_candidate_per_model,
#                         )
#                     )
#                     augmented_action_ids_ = self.target_late_stage_policy.sample(
#                         context=context_,
#                         context_id=context_id_,
#                         latent=latent_,
#                         latent_id=latent_id_,
#                         candidate_actions=candidate_actions_,
#                         n_output_action=action_ids_.shape[1],
#                         factual_rewards=factual_rewards_,
#                     )

#                     action_ = self.action_embedding_model.retrieve(  # pyre-ignore
#                         action_ids=action_ids_,
#                         is_raw_embedding=self.use_raw_action_embedding,
#                     )
#                     augmented_action_ = (
#                         self.action_embedding_model.retrieve(  # pyre-ignore
#                             action_ids=augmented_action_ids_,
#                             is_raw_embedding=self.use_raw_action_embedding,
#                         )
#                     )

#                     # currently only context-free kernel function is supported
#                     # thus, the context info is ignored by the default kernel implementation
#                     action_dist_ = self.distance_model(
#                         context=context_,
#                         context_id=context_id_,
#                         latent=latent_,
#                         latent_id=latent_id_,
#                         action=action_,
#                         augmented_action=augmented_action_,
#                     )
#                     kernel_weight_ = kernel_function(
#                         action_dist_,
#                         **kernel_function_kwargs,
#                     )

#                     loss_ = self._policy_gradient(
#                         early_stage_log_prob=early_stage_log_prob_,
#                         kernel_weight=kernel_weight_,
#                         logging_marginal_prob=logging_marginal_prob_,
#                         agg_reward=agg_reward_,
#                     )

#                     optimizer.zero_grad()
#                     loss_.backward()
#                     optimizer.step()

#                     train_losses[i + 1] += loss_ / n_steps_per_epoch

#                 if (i + 1) % n_epochs_per_log == 0:
#                     with torch.no_grad():
#                         for batch_ in val_loader:
#                             context_ = batch_["context"].to(
#                                 self.device, non_blocking=True
#                             )
#                             latent_ = batch_["latent"].to(
#                                 self.device, non_blocking=True
#                             )
#                             context_id_ = batch_["context_id"].to(
#                                 self.device, non_blocking=True
#                             )
#                             latent_id_ = batch_["latent_id"].to(
#                                 self.device, non_blocking=True
#                             )
#                             action_ids_ = batch_["action"].to(
#                                 self.device, non_blocking=True
#                             )
#                             agg_reward_ = batch_["agg_reward"].to(
#                                 self.device, non_blocking=True
#                             )
#                             factual_rewards_ = batch_["factual_rewards"].to(
#                                 self.device, non_blocking=True
#                             )

#                             with torch.no_grad():
#                                 logging_marginal_prob_ = logging_marginal_model(
#                                     context=context_,
#                                     context_id=context_id_,
#                                     latent=latent_,
#                                     latent_id=latent_id_,
#                                     actions=action_ids_,
#                                 )

#                             candidate_actions_, _, early_stage_log_prob_ = (
#                                 early_stage_policy.sample_actions_with_prob(
#                                     context=context_,
#                                     context_id=context_id_,
#                                     n_candidate_action=n_candidate_action,
#                                 )
#                             )
#                             augmented_action_ids_ = (
#                                 self.target_late_stage_policy.sample(
#                                     context=context_,
#                                     context_id=context_id_,
#                                     latent=latent_,
#                                     latent_id=latent_id_,
#                                     candidate_actions=candidate_actions_,
#                                     factual_rewards=factual_rewards_,
#                                     n_output_action=action_ids_.shape[1],
#                                 )
#                             )

#                             action_ = (
#                                 self.action_embedding_model.retrieve(  # pyre-ignore
#                                     action_ids=action_ids_,
#                                     is_raw_embedding=self.use_raw_action_embedding,
#                                 )
#                             )
#                             augmented_action_ = (
#                                 self.action_embedding_model.retrieve(  # pyre-ignore
#                                     action_ids=augmented_action_ids_,
#                                     is_raw_embedding=self.use_raw_action_embedding,
#                                 )
#                             )
#                             action_dist_ = self.distance_model(
#                                 context=context_,
#                                 context_id=context_id_,
#                                 latent=latent_,
#                                 latent_id=latent_id_,
#                                 action=action_,
#                                 augmented_action=augmented_action_,
#                             )
#                             kernel_weight_ = kernel_function(
#                                 action_dist_,
#                                 **kernel_function_kwargs,
#                             )
#                             val_loss_ = self._policy_gradient(
#                                 early_stage_log_prob=early_stage_log_prob_,
#                                 kernel_weight=kernel_weight_,
#                                 logging_marginal_prob=logging_marginal_prob_,
#                                 agg_reward=agg_reward_,
#                             )

#                             val_losses[(i + 1) // n_epochs_per_log] += (
#                                 val_loss_ * len(batch_["context_id"]) / len(val_dataset)
#                             )

#                     if self.env is not None:
#                         policy_value_ = self.env.evaluate_policy_online(  # pyre-ignore
#                             early_stage_policy=early_stage_policy,
#                             late_stage_policy=self.eval_late_stage_policy,
#                             is_deterministic_early_stage=is_deterministic_early_stage_eval,
#                             is_deterministic_late_stage=is_deterministic_late_stage_eval,
#                             n_candidate_action=n_candidate_action,
#                             n_candidate_per_model=n_candidate_per_model,
#                         )
#                         policy_values[(i + 1) // n_epochs_per_log] += (
#                             policy_value_.item()
#                         )

#                     # early stopping
#                     current_epoch = (i + 1) // n_epochs_per_log
#                     best_loss_epoch = torch.argmin(
#                         val_losses[1 : current_epoch + 1]
#                     ).item()

#                     if current_epoch - best_loss_epoch > patience:  # pyre-ignore
#                         print("early stopping at epoch", i + 1)
#                         break

#         self.trained_early_stage_policy = early_stage_policy  # pyre-ignore

#         if save_path is not None:
#             self.save_early_stage_model(save_path)

#         if return_training_logs:
#             output = (
#                 early_stage_policy,
#                 {
#                     "train_losses": train_losses,
#                     "val_losses": val_losses,
#                     "policy_values": policy_values,
#                 },
#             )
#         else:
#             output = early_stage_policy

#         return output
