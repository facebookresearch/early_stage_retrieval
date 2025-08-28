# pyre-unsafe
"""Implementation of the greedy subset selection algorithm for the early stage policy."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import BaseEarlyStagePolicy


@dataclass
class GreedySubsetEarlyStagePolicy(BaseEarlyStagePolicy):
    """Greedy Subset Selection algorithm for early stage retrieval.

    Input
    ------
    base_quantile_model: nn.Module
        The base model for quantile regression.

    action_set: BaseActionSet
        The action set for the policy.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_quantile_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_quantile_model = self.base_quantile_model.to(self.device)

        self.n_action = self.action_set.n_action

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_output_action: int = 1,
        n_monte_carlo_sample: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the top-k actions.

        Input
        ------
        context: Optional[torch.Tensor]
            The context tensor.

        context_id: Optional[torch.Tensor]
            The context id tensor.

        n_candidate_action: int
            The number of candidate actions.

        n_output_action: int
            The number of output actions.

        n_monte_carlo_sample: int
            The number of Monte Carlo samples.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        n_sample = (
            context.shape[0]
            if context is not None
            else context_id.shape[0]  # pyre-ignore
        )
        idxes = torch.arange(n_sample, device=self.device)
        batch_idxes = idxes.unsqueeze(-1).expand(-1, n_monte_carlo_sample)
        mc_idxes = (
            torch.arange(n_monte_carlo_sample, device=self.device)
            .unsqueeze(0)
            .expand(n_sample, -1)
        )

        target_quartile = torch.rand(
            (
                n_sample,
                self.n_action,  # pyre-ignore
                n_monte_carlo_sample,
            ),
            device=self.device,
        )
        sampled_values = self.base_quantile_model(
            context=context,
            context_id=context_id,
            target_quantile=target_quartile,
        )  # (n_sample, n_action, n_monte_carlo_sample)

        # initialize the variables
        lowest_value = torch.zeros(
            (n_sample, n_monte_carlo_sample),
            device=self.device,
        )
        subset_values = torch.full(
            (n_sample, n_output_action, n_monte_carlo_sample),
            torch.inf,
            device=self.device,
        )

        selected_actions = []
        for k in range(n_candidate_action):
            if k < n_output_action:
                values_ = sampled_values
            else:
                # lowest_value refers to the lowest value among the top-(n_output_action)
                values_ = torch.clip(
                    sampled_values - lowest_value.unsqueeze(1), min=0.0
                )

            _, next_action_ = values_.mean(dim=-1).max(dim=1)
            next_value_ = sampled_values[idxes, next_action_, :]

            # update the record (on-going)
            if k == 0:
                lowest_value = next_value_
                subset_values[:, 0, :] = sampled_values[idxes, next_action_, :]
                lowest_idx = torch.zeros(
                    (n_sample, n_monte_carlo_sample),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                lowest_value = torch.min(lowest_value, next_value_)
                subset_values[batch_idxes, lowest_idx, mc_idxes] = (  # pyre-ignore
                    lowest_value
                )
                _, lowest_idx = subset_values.min(dim=1)

            # update the value so that the already selectede actions are not selected again
            sampled_values[idxes, next_action_, :] = -torch.inf
            selected_actions.append(next_action_.unsqueeze(-1))

        selected_actions = torch.cat(selected_actions, dim=-1)
        return selected_actions

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_output_action: int = 1,
        n_monte_carlo_sample: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        n_output_action: int
            The number of output actions to sample.

        n_monte_carlo_sample: int
            The number of Monte Carlo samples.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions (non-differential).

        """
        actions = self.retrieve_topk(
            context=context,
            context_id=context_id,
            n_candidate_action=n_candidate_action,
            n_output_action=n_output_action,
            n_monte_carlo_sample=n_monte_carlo_sample,
        )
        return actions

    def sample_actions_with_prob(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_output_action: int = 1,
        n_monte_carlo_sample: int = 100,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        n_output_action: int
            The number of output actions to sample.

        n_monte_carlo_sample: int
            The number of Monte Carlo samples.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        logprob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, non-differential).

        """
        actions = self.retrieve_topk(
            context=context,
            context_id=context_id,
            n_candidate_action=n_candidate_action,
            n_output_action=n_output_action,
            n_monte_carlo_sample=n_monte_carlo_sample,
        )
        prob = torch.ones((actions.shape[0],), device=actions.device)
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return actions, prob, log_prob

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        n_monte_carlo_sample: int = 100,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to calculate the probability of the given actions. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to calculate the probability of the given actions. Either context or context_id must be provided.

        n_output_action: int
            The number of output actions.

        n_monte_carlo_sample: int
            The number of Monte Carlo samples.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        greedy_actions = self.retrieve_topk(
            context=context,
            context_id=context_id,
            n_candidate_action=actions.shape[-1],
            n_output_action=n_output_action,
            n_monte_carlo_sample=n_monte_carlo_sample,
        )

        prob = torch.all(actions == greedy_actions, dim=1).float()
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return prob, log_prob
