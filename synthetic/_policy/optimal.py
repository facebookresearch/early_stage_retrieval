# pyre-unsafe
"""Implementation of the optimal early and late stage policies."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import BaseEarlyStagePolicy, BaseLateStagePolicy, BaseSingleStagePolicy


@dataclass
class OptimalEarlyStagePolicy(BaseEarlyStagePolicy):
    """Implementation of the optimal early stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action

            self.n_action = self.action_set.n_action

    def retrieve_topk(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        _, topk = torch.topk(factual_rewards, k=n_candidate_action, dim=1)
        return topk

    def sample(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_candidate_action=n_candidate_action,
        )
        return actions

    def sample_actions_with_prob(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        logprob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, non-differential).

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_candidate_action=n_candidate_action,
        )
        prob, log_prob = self.calc_prob_given_actions(
            factual_rewards=factual_rewards,
            actions=actions,
        )
        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to calculate the probability of the given actions. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to calculate the probability of the given actions. Either context or context_id must be provided.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        optimal_actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_candidate_action=actions.shape[1],
        )

        prob = torch.all(actions == optimal_actions, dim=1).float()
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return prob, log_prob


@dataclass
class OptimalLateStagePolicy(BaseLateStagePolicy):
    """Implementation of the optimal late stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action
            self.n_action = self.action_set.n_action

    def retrieve_topk(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to retrieve from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)
        _, within_candidate_idx = torch.topk(factual_rewards, k=n_output_action, dim=1)
        topk = torch.gather(candidate_actions, 1, within_candidate_idx)
        return topk

    def sample(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            candidate_actions=candidate_actions,
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        return actions

    def sample_actions_with_prob(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        logprob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, non-differential).

        """
        actions = self.retrieve_topk(
            candidate_actions=candidate_actions,
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        prob = torch.ones((actions.shape[0],), device=actions.device)
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        optimal_actions = self.retrieve_topk(
            candidate_actions=candidate_actions,
            factual_rewards=factual_rewards,
            n_output_action=actions.shape[1],
        )

        prob = torch.all(actions == optimal_actions, dim=1).float()
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return prob, log_prob


@dataclass
class OptimalSingleStagePolicy(BaseSingleStagePolicy):
    """Implementation of the optimal single stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        The base model.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action
            self.n_action = self.action_set.n_action

    def retrieve_topk(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        _, topk = torch.topk(factual_rewards, k=n_output_action, dim=1)
        return topk

    def sample(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        return actions

    def sample_actions_with_prob(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        logprob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, non-differential).

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        prob = torch.ones((actions.shape[0],), device=actions.device)
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        optimal_actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_output_action=actions.shape[1],
        )

        prob = torch.all(actions == optimal_actions, dim=1).float()
        log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        return prob, log_prob


@dataclass
class OracleSoftmaxEarlyStagePolicy(BaseEarlyStagePolicy):
    """Early stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action

            self.n_action = self.action_set.n_action

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        logits: torch.Tensor,
        n_candidate_action: int = 1,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        _, topk = torch.topk(logits, k=n_candidate_action, dim=1)
        return topk

    def _log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_actions)
            The actions to calculate the probability.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_sample, n_candidate_action = actions.shape
        idxes = torch.arange(n_sample, device=self.device)

        mask = torch.ones_like(logits)
        idxes = torch.arange(n_sample, device=actions.device)
        log_prob = torch.gather(logits, 1, actions).sum(dim=-1)  # (n_samples, )

        for k in range(n_candidate_action):
            remaining_logits = torch.masked_select(logits, mask.bool()).view(
                (n_sample, -1)
            )  # (n_samples, n_actions - k)
            log_prob = log_prob - torch.logsumexp(remaining_logits, dim=1)

            # .clone() is needed for the gradient to propagate (mask.scatter_ did not work)
            mask = mask.clone()
            mask[idxes, actions[:, k]] = 0

        return log_prob

    def retrieve_topk(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        topk = self._topk(
            logits=factual_rewards,
            n_candidate_action=n_candidate_action,
        )
        return topk

    def sample(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(  # pyre-ignore
                factual_rewards.shape
            ).to(self.device)

        actions = self._topk(
            logits=factual_rewards + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
        )
        return actions

    def sample_actions_with_prob(  # pyre-ignore
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with the plackett-luce probability.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(  # pyre-ignore
                factual_rewards.shape
            ).to(self.device)

        actions = self._topk(
            logits=factual_rewards + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
        )

        if is_deterministic:
            prob = torch.ones((actions.shape[0],), device=self.device)
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=factual_rewards,
                actions=actions,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        is_deterministic: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the plackett-luce probability given actions.

        Input
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        factual_rewards: Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        if is_deterministic:
            selected_actions = self.retrieve_topk(
                factual_rewards=factual_rewards,
                context=context,
                context_id=context_id,
                n_candidate_action=actions.shape[1],
            )
            prob = torch.all(actions == selected_actions, dim=1).float()
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)

        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=factual_rewards,
                actions=actions,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob


@dataclass
class OracleSoftmaxLateStagePolicy(BaseLateStagePolicy):
    """Late stage ranking/recommendation policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action

            self.n_action = self.action_set.n_action

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        candidate_actions: torch.Tensor,
        logits: torch.Tensor,
        n_output_action: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to retrieve from.

        logits: torch.Tensor, shape (n_samples, n_candidate_action)
            The logits of the actions.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        within_candidate_idx: torch.Tensor, shape (n_samples, n_output_action)
            The index of the sampled actions within the candidate actions.

        """
        _, within_candidate_idx = torch.topk(logits, k=n_output_action, dim=1)
        actions = torch.gather(candidate_actions, 1, within_candidate_idx)
        return actions, within_candidate_idx

    def _log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        within_candidate_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Input
        ------
        logit: torch.Tensor, shape (n_samples, n_candidate_actions)
            The logits of the actions.

        within_candidate_idx: torch.Tensor, shape (n_samples, n_output_action)
            The index of the actions within the candidate actions.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_sample, n_output_action = within_candidate_idx.shape

        mask = torch.ones_like(logits)
        idxes = torch.arange(n_sample, device=within_candidate_idx.device)
        log_prob = torch.gather(logits, 1, within_candidate_idx).sum(
            dim=-1
        )  # (n_samples, )

        for k in range(n_output_action):
            remaining_logits = torch.masked_select(logits, mask.bool()).view(
                n_sample, -1
            )  # (n_samples, n_candidate_action - k)
            log_prob = log_prob - torch.logsumexp(remaining_logits, dim=1)

            # .clone() is needed for the gradient to propagate (mask.scatter_ did not work)
            mask = mask.clone()
            mask[idxes, within_candidate_idx[:, k]] = 0

        return log_prob

    def retrieve_topk(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to retrieve from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=factual_rewards,
            n_output_action=n_output_action,
        )
        return topk

    def sample(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor
            The sampled actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(  # pyre-ignore
                factual_rewards.shape
            ).to(self.device)

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=factual_rewards + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk

    def sample_actions_with_prob(  # pyre-ignore
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(  # pyre-ignore
                factual_rewards.shape
            ).to(self.device)

        actions, within_candidate_idx = self._topk(
            candidate_actions=candidate_actions,
            logits=factual_rewards + gumbel_noise,
            n_output_action=n_output_action,
        )

        if is_deterministic:
            prob = torch.ones((actions.shape[0],), device=self.device)
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=factual_rewards, within_candidate_idx=within_candidate_idx
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        is_deterministic: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        if is_deterministic:
            selected_actions = self.retrieve_topk(
                candidate_actions=candidate_actions,
                factual_rewards=factual_rewards,
                n_candidate_action=actions.shape[1],
            )
            prob = torch.all(actions == selected_actions, dim=1).float()
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)

        else:
            # identify the within_candidate_idx
            candidate_actions = candidate_actions.unsqueeze(1)
            actions = actions.unsqueeze(2)
            mask = candidate_actions == actions
            within_candidate_idx = mask.float().argmax(dim=2)

            log_prob = self._log_prob_given_logits_and_actions(
                logits=factual_rewards, within_candidate_idx=within_candidate_idx
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob
