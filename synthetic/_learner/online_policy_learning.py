"""Traning procedure of on-policy learning."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from early_stage_retrieval.synthetic.dataset import (
    BaseDataGenerator,
)
from early_stage_retrieval.synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
)
from torch.optim import Adagrad, Optimizer
from tqdm.auto import tqdm

from .base import BasePolicyLearner


@dataclass
class OnlinePolicyLearner(BasePolicyLearner):
    def train_early_stage_policy_online(self, **kwargs):
        raise NotImplementedError()

    def train_early_stage_policy_offline(self, **kwargs):
        raise NotImplementedError()


# @dataclass
# class OnlinePolicyLearner(BasePolicyLearner):
#     """Training procedure of on-policy learning.

#     Input
#     ------
#     env: BaseDataGenerator
#         The data generation environment.

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

#     early_stage_optimizer: Optimizer = Adagrad
#         The optimizer to use for the early stage policy.

#     early_stage_optimizer_kwargs: Optional[Dict[str, Any]]
#         The optimizer kwargs to use for the early stage policy.

#     device: torch.device, default=torch.device("cpu")
#         The device to use.

#     random_seed: Optional[int]
#         The random seed to use.

#     """

#     env: BaseDataGenerator
#     early_stage_policy: BaseEarlyStagePolicy
#     target_late_stage_policy: BaseLateStagePolicy
#     eval_late_stage_policy: BaseLateStagePolicy
#     is_model_free_target_late_stage_policy: bool = False
#     is_model_free_eval_late_stage_policy: bool = False
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
#         agg_reward: torch.Tensor,
#     ):
#         """Compute the policy gradient.

#         Input
#         ------
#         early_stage_log_prob: torch.Tensor, shape (n_samples, )
#             The log probability of the candidate action (differential).

#         agg_reward: torch.Tensor, shape (n_samples, )
#             The aggregated reward (non-differential).

#         Output
#         ------
#         policy_gradient: torch.Tensor, shape (1, )
#             The policy gradient.

#         """
#         policy_gradient = -early_stage_log_prob * agg_reward
#         return policy_gradient.mean()

#     def train_early_stage_policy_online(
#         self,
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

#         train_losses = torch.zeros((n_epoch + 1,), device=self.device)
#         action_probs = torch.zeros((n_epoch + 1,), device=self.device)
#         policy_values = torch.zeros(
#             (n_epoch // n_epochs_per_log + 1,), device=self.device
#         )

#         with tqdm(range(n_epoch)) as pbar:
#             for i, ch in enumerate(pbar):
#                 pbar.set_description(f"[Train base early stage model: Epoch {i}]")
#                 pbar.set_postfix(
#                     {
#                         "train_loss": f"{train_losses[i]:.4g}",
#                         "action_prob": f"{action_probs[i]:.4g}",
#                         "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
#                     },
#                 )

#                 for j in range(n_steps_per_epoch):
#                     context_, context_id_ = (
#                         self.env.context_sampler.sample(  # pyre-ignore
#                             batch_size
#                         )
#                     )
#                     latent_, latent_id_ = self.env.latent_sampler.sample(  # pyre-ignore
#                         batch_size
#                     )

#                     factual_rewards_ = self.env.retrieve_factual_rewards(  # pyre-ignore
#                         context_id=context_id_,
#                         latent_id=latent_id_,
#                     )

#                     candidate_actions_, early_stage_prob_, early_stage_log_prob_ = (
#                         early_stage_policy.sample_actions_with_prob(
#                             context=context_,
#                             context_id=context_id_,
#                             n_candidate_action=n_candidate_action,
#                             n_candidate_per_model=n_candidate_per_model,
#                         )
#                     )
#                     action_ids_ = self.target_late_stage_policy.sample(
#                         context=context_,
#                         context_id=context_id_,
#                         latent=latent_,
#                         latent_id=latent_id_,
#                         candidate_actions=candidate_actions_,
#                         factual_rewards=factual_rewards_,
#                     )
#                     _, agg_reward_ = self.env.reward_model.sample(  # pyre-ignore
#                         context=context_,
#                         latent=latent_,
#                         actions=action_ids_,
#                     )

#                     loss_ = self._policy_gradient(
#                         early_stage_log_prob=early_stage_log_prob_,
#                         agg_reward=agg_reward_,
#                     )

#                     optimizer.zero_grad()

#                     with torch.autograd.set_detect_anomaly(True):
#                         loss_.backward()

#                     optimizer.step()

#                     train_losses[i + 1] += loss_ / n_steps_per_epoch
#                     action_probs[i + 1] += (  # pyre-ignore
#                         early_stage_prob_  # pyre-ignore
#                         ** (1 / self.env.reward_model.n_output_action)
#                     ).mean() / n_steps_per_epoch

#                 if (i + 1) % n_epochs_per_log == 0:
#                     policy_value_ = self.env.evaluate_policy_online(  # pyre-ignore
#                         early_stage_policy=early_stage_policy,
#                         late_stage_policy=self.eval_late_stage_policy,
#                         is_deterministic_early_stage=is_deterministic_early_stage_eval,
#                         is_deterministic_late_stage=is_deterministic_late_stage_eval,
#                         n_candidate_action=n_candidate_action,
#                         n_candidate_per_model=n_candidate_per_model,
#                     )
#                     policy_values[(i + 1) // n_epochs_per_log] += policy_value_.item()

#         self.trained_early_stage_policy = early_stage_policy  # pyre-ignore

#         if save_path is not None:
#             self.save_early_stage_model(save_path)

#         if return_training_logs:
#             output = (
#                 early_stage_policy,
#                 {
#                     "train_losses": train_losses,
#                     "action_probs": action_probs,
#                     "policy_values": policy_values,
#                 },
#             )
#         else:
#             output = early_stage_policy

#         return output

#     def train_early_stage_policy_offline(self, **kwargs):
#         raise NotImplementedError()
