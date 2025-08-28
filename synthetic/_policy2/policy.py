"""Policy class for the early stage retrieval and late stage ranking policies."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import (
    BaseEarlyStagePolicy,
    BaseJointPolicy,
    BaseLateStagePolicy,
    BaseSingleStagePolicy,
)


@dataclass
class BaselineEarlyStagePolicy(BaseEarlyStagePolicy):
    """Early stage retrieval policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _n_candidate_per_model(
        self,
        n_candidate_action: int,
        n_model: int,
    ):
        """Retrieve the number of candidate actions per model.

        Input
        ------
        n_candidate_action: int
            The number of candidate actions.

        n_model: int
            The number of models.

        Output
        ------
        assignment: torch.Tensor, shape (n_model, )
            The number of candidate actions per model.

        """
        base = n_candidate_action // n_model
        remainder = n_candidate_action % n_model
        assignment = [base + 1 if i < remainder else base for i in range(n_model)]
        return torch.tensor(assignment, device=self.device)

    def _topk(
        self,
        logits: torch.Tensor,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        n_model = logits.shape[0]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        candidate_actions = []
        model_idx = torch.arange(n_model, device=self.device).view(-1, 1, 1)
        sample_idx = torch.arange(len(logits[0]), device=self.device).view(1, -1, 1)
        for model_id_ in range(n_model):
            _, candidate_ = torch.topk(
                logits[model_id_], k=n_candidate_per_model[model_id_], dim=1
            )
            logits[model_idx, sample_idx, candidate_.unsqueeze(0)] = -torch.inf
            candidate_actions.append(candidate_)

        return torch.cat(candidate_actions, dim=1)

    def _log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_model, n_sample, _ = logits.shape
        n_candidate_action = actions.shape[1]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        model_id = torch.tensor(
            [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
            device=self.device,
        )

        """Straightforward implementation (slow)"""
        # log_prob = 0
        # mask = torch.ones_like(logits[0])
        # idxes = torch.arange(n_sample, device=self.device)

        # for k, model_id_ in enumerate(model_id):
        #     remaining_logits = torch.masked_select(logits[model_id_], mask.bool()).view(
        #         n_sample, -1
        #     )  # (n_samples, n_candidate_action - k)
        #     log_prob = (
        #         log_prob
        #         + torch.gather(
        #             logits[model_id_], 1, actions[:, k].unsqueeze(1)
        #         ).squeeze(1)
        #         - torch.logsumexp(remaining_logits, dim=1)
        #     )

        #     # .clone() is needed for the gradient to propagate (mask.scatter_ did not work)
        #     mask = mask.clone()
        #     mask[idxes, actions[:, k]] = 0

        """Alternative implementation (fast)

        For each position, the log probability is given by:
            log_prob = logit(kth selected) - A - log1p(-exp(B-A))

        where A = logsumexp( logit(all) ) and B = logsumexp( logit(selected by kth) )"""

        if require_grad_model_id is None:
            # first calculate the logits(kth selected) (n_samples, n_candidate_action)
            sample_indices = torch.arange(n_sample, device=self.device).unsqueeze(1)
            selected_logits = logits[model_id.unsqueeze(0), sample_indices, actions]
            logits_selected_sum = torch.sum(selected_logits, dim=1)  # (n_samples, )

            # next, we calculate A = logsumexp( logit(all) )
            logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_model, n_samples)
            logsumexp_all = logsumexp_all[model_id].T  # (n_samples, n_candidate_action)
            logsumexp_all_sum = torch.sum(logsumexp_all, dim=1)  # (n_samples, )

            # then, we calculate B = logsumexp( logit(selected by kth) )
            selected_mask = torch.tril(
                torch.ones((n_candidate_action, n_candidate_action), device=self.device)
            ) - torch.eye(n_candidate_action, device=self.device)

            logits_topk = torch.gather(
                logits[model_id],
                2,
                actions.unsqueeze(0).expand(n_candidate_action, -1, -1),
            )  # (n_candidate_action, n_samples, n_candidate_action)

            # avoiding this for-loop produced NaN values a lot, so I kept this part
            if n_candidate_action == 1:
                logsumexp_selected = 0
            else:
                logsumexp_selected = []
                for k in range(1, n_candidate_action):
                    selected_logits = torch.masked_select(
                        logits_topk[k], selected_mask[k].unsqueeze(0).bool()
                    ).reshape(n_sample, -1)  # (n_samples, n_candidate_action - k)
                    logsumexp_selected.append(
                        torch.logsumexp(selected_logits, dim=1).unsqueeze(1)
                    )  # (n_samples, )

                logsumexp_selected = torch.cat(
                    logsumexp_selected, dim=1
                )  # (n_samples, n_candidate_action - 1)

            # finally, we calculate log1p(-exp(B-A))
            # Note that we need to exclude the first position as no action is selected at the beggining.
            log1p_selected = torch.log1p(
                -torch.exp(logsumexp_selected - logsumexp_all[:, 1:])
            )  # (n_samples, n_candidate_action - 1)
            log1p_selected_sum = torch.sum(log1p_selected, dim=1)  # (n_samples, )

            # summing up each different terms
            log_prob = logits_selected_sum - logsumexp_all_sum - log1p_selected_sum

        else:
            # first calculate the logits(kth selected) (n_samples, n_candidate_action)
            candidate_start_idx = sum(n_candidate_per_model[:require_grad_model_id])
            candidate_end_idx = (
                candidate_start_idx + n_candidate_per_model[require_grad_model_id]
            )
            sample_indices = torch.arange(n_sample, device=self.device).unsqueeze(1)
            selected_logits = logits[require_grad_model_id][
                sample_indices, actions[:, candidate_start_idx:candidate_end_idx]
            ]
            logits_selected_sum = torch.sum(selected_logits, dim=1)  # (n_samples, )

            # next, we calculate A = logsumexp( logit(all) )
            logsumexp_all = torch.logsumexp(
                logits[require_grad_model_id], dim=1
            )  # (n_samples, )
            logsumexp_all_sum = (
                logsumexp_all * n_candidate_per_model[require_grad_model_id]
            )  # (n_samples, )

            # then, we calculate B = logsumexp( logit(selected by kth) )
            selected_mask = torch.tril(
                torch.ones((n_candidate_action, n_candidate_action), device=self.device)
            ) - torch.eye(n_candidate_action, device=self.device)

            logits_topk = torch.gather(
                logits[require_grad_model_id],
                1,
                actions,
            )  # (n_samples, n_candidate_action)

            # avoiding this for-loop produced NaN values a lot, so I kept this part
            if n_candidate_action == 1:
                logsumexp_selected = 0
            else:
                logsumexp_selected = []
                for k in range(min(1, candidate_start_idx), candidate_end_idx):
                    selected_logits = torch.masked_select(
                        logits_topk, selected_mask[k].bool()
                    ).reshape(n_sample, -1)  # (n_samples, n_candidate_action - k)
                    logsumexp_selected.append(
                        torch.logsumexp(selected_logits, dim=1).unsqueeze(1)
                    )  # (n_samples, )

                logsumexp_selected = torch.cat(
                    logsumexp_selected, dim=1
                )  # (n_samples, n_candidate_per_model[require_grad_model_id] (- 1))

            # finally, we calculate log1p(-exp(B-A))
            log1p_selected = torch.log1p(
                -torch.exp(logsumexp_selected - logsumexp_all.unsqueeze(1))
            )
            log1p_selected_sum = torch.sum(log1p_selected, dim=1)  # (n_samples, )

            # summing up each different terms
            log_prob = logits_selected_sum - logsumexp_all_sum - log1p_selected_sum

        return log_prob  # pyre-ignore

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(context=context, context_id=context_id)

        topk = self._topk(
            logits=logits,
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )
        return topk

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(context=context, context_id=context_id)

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        actions = self._topk(
            logits=logits + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )
        return actions

    def sample_actions_with_prob(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic: bool = False,
        require_grad_model_id: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with the plackett-luce probability.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        is_deterministic: bool
            Whether to use deterministic sampling.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        logits = self.base_model(
            context=context,
            context_id=context_id,
            require_grad_model_id=require_grad_model_id,
        )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        actions = self._topk(
            logits=logits + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )

        if is_deterministic:
            prob = torch.ones((actions.shape[0],), device=self.device)
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits,
                actions=actions,
                n_candidate_per_model=n_candidate_per_model,
                require_grad_model_id=require_grad_model_id,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return actions, prob, log_prob

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic: bool = False,
        require_grad_model_id: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the plackett-luce probability given actions.

        Input
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        is_deterministic: bool
            Whether to use deterministic sampling.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        if is_deterministic:
            selected_actions = self.retrieve_topk(
                context=context,
                context_id=context_id,
                n_candidate_action=actions.shape[1],
                n_candidate_per_model=n_candidate_per_model,
            )
            prob = torch.all(actions == selected_actions, dim=1).float()
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)

        else:
            logits = self.base_model(context=context, context_id=context_id)
            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits,
                actions=actions,
                n_candidate_per_model=n_candidate_per_model,
                require_grad_model_id=require_grad_model_id,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob


@dataclass
class BaselineLateStagePolicy(BaseLateStagePolicy):
    """Late stage ranking/recommendation policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

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
        logits: torch.Tensor, shape (n_samples, n_candidate_actions)
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
        idxes = torch.arange(n_sample, device=self.device)

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

        # ### Alternative implementation
        # # For each position, the log probability is given by:
        # #     log_prob = logit(kth selected) - A - log1p(-exp(B-A))

        # # # where A = logsumexp( logit(all) ) and B = logsumexp( logit(selected by kth) )

        # # first calculate the logits(kth selected) (n_samples, n_candidate_action)
        # sample_indices = torch.arange(n_sample, device=self.device)
        # selected_logits = logits[sample_indices, within_candidate_idx]
        # logits_selected_sum = torch.sum(selected_logits, dim=1)  # (n_samples, )

        # # next, we calculate A = logsumexp( logit(all) )
        # logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_samples, )
        # logsumexp_all_sum = logsumexp_all * n_output_action

        # # then, we calculate B = logsumexp( logit(selected by kth) )
        # selected_mask = torch.tril(
        #     torch.ones((n_output_action, n_output_action), device=self.device)
        # )

        # logits_topk = torch.gather(
        #     logits, 1, within_candidate_idx
        # )  # (n_samples, n_output_action)
        # exp_logits_topk = torch.exp(logits_topk)

        # logsumexp_selected = torch.log(
        #     torch.sum(exp_logits_topk.unsqueeze(0) * selected_mask.unsqueeze(1), dim=2)
        # ).T  # (n_samples, n_output_action)

        # # finally, we calculate log1p(-exp(B-A))
        # # Note that we need to exclude the first position as no action is selected at the beggining.
        # log1p_selected = torch.log1p(
        #     -torch.exp(logsumexp_selected[:, :-1] - logsumexp_all.unsqueeze(1))
        # )  # (n_samples, n_output_action - 1)
        # log1p_selected_sum = torch.sum(log1p_selected, dim=1)  # (n_samples, )

        # log_prob = logits_selected_sum - logsumexp_all_sum - log1p_selected_sum

        return log_prob

    def retrieve_topk(
        self,
        candidate_actions: torch.Tensor,
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

        context: Tensor, shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                action_ids=candidate_actions,
            )

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=logits,
            n_output_action=n_output_action,
        )
        return topk

    def sample(
        self,
        candidate_actions: torch.Tensor,
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

        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                action_ids=candidate_actions,
            )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk

    def sample_actions_with_prob(
        self,
        candidate_actions: torch.Tensor,
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

        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

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
        logits = self.base_model(
            context=context,
            context_id=context_id,
            latent=latent,
            latent_id=latent_id,
            action_ids=candidate_actions,
        )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        actions, within_candidate_idx = self._topk(
            candidate_actions=candidate_actions,
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )

        if is_deterministic:
            prob = torch.ones((actions.shape[0],), device=self.device)
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits, within_candidate_idx=within_candidate_idx
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return actions, prob, log_prob

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        candidate_actions: torch.Tensor,
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
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

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
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                n_candidate_action=actions.shape[1],
            )
            prob = torch.all(actions == selected_actions, dim=1).float()
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)

        else:
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                action_ids=candidate_actions,
            )

            # identify the within_candidate_idx
            candidate_actions = candidate_actions.unsqueeze(1)
            actions = actions.unsqueeze(2)
            mask = candidate_actions == actions
            within_candidate_idx = mask.float().argmax(dim=2)

            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits, within_candidate_idx=within_candidate_idx
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob


@dataclass
class BaselineJointPolicy(BaseJointPolicy):
    """Joint (early stage + late stage) recommendation/ranking policy.

    Input
    ------
    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    late_stage_policy: BaseLateStagePolicy
        The late stage policy.

    action_set: BaseActionSet
        The action set.

    is_deterministic_early_stage: bool
        Whether the early stage policy is deterministic.

    is_deterministic_late_stage: bool
        Whether the late stage policy is deterministic.

    is_model_free_early_stage: bool
        Whether the early stage policy is model free.

    is_model_free_late_stage: bool
        Whether the late stage policy is model free.

    n_candidate_action: int
        The number of candidate actions.

    n_output_action: int
        The number of output actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    early_stage_policy: BaseEarlyStagePolicy
    late_stage_policy: BaseLateStagePolicy
    action_set: BaseActionSet
    is_deterministic_early_stage: bool = False
    is_deterministic_late_stage: bool = False
    is_model_free_early_stage: bool = False
    is_model_free_late_stage: bool = False
    n_candidate_action: int = 1
    n_output_action: int = 1
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

            if not self.is_model_free_early_stage:
                self.early_stage_policy.base_model = (
                    self.early_stage_policy.base_model.to(self.device)
                )
            else:
                self.late_stage_policy.base_model = (
                    self.late_stage_policy.base_model.to(self.device)
                )

            self.action_prob_regressor = self.action_prob_regressor.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        candidate_actions = self.early_stage_policy.retrieve_topk(
            context=context,
            context_id=context_id,
            n_candidate_action=self.n_candidate_action,
            n_output_action=self.n_output_action,
            is_deterministic=self.is_deterministic_early_stage,
        )
        actions = self.late_stage_policy.retrieve_topk(
            candidate_actions=candidate_actions,
            context=context,
            context_id=context_id,
            latent=latent,
            latent_id=latent_id,
            n_output_action=self.n_output_action,
            is_deterministic=self.is_deterministic_late_stage,
        )
        return actions

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        if self.is_deterministic_early_stage:
            candidate_actions = self.early_stage_policy.retrieve_topk(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                n_output_action=self.n_output_action,  # only when using greedy algorithm
                is_deterministic=self.is_deterministic_early_stage,
            )
        else:
            candidate_actions = self.early_stage_policy.sample(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                is_deterministic=self.is_deterministic_early_stage,
            )
        if self.is_deterministic_late_stage:
            actions = self.late_stage_policy.retrieve_topk(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )
        else:
            actions = self.late_stage_policy.sample(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )
        return actions

    def sample_actions_with_prob(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample k-candidate actions with probability.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions (non-differential).

        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        logprob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        if self.is_deterministic_early_stage:
            candidate_actions = self.early_stage_policy.retrieve_topk(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                n_output_action=self.n_output_action,  # only when using greedy algorithm
                is_deterministic=self.is_deterministic_early_stage,
            )
        else:
            candidate_actions = self.early_stage_policy.sample(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                is_deterministic=self.is_deterministic_early_stage,
            )

        if self.is_deterministic_late_stage:
            actions = self.late_stage_policy.retrieve_topk(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )
        else:
            actions = self.late_stage_policy.sample(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )

        prob, log_prob = self.calc_prob_given_actions(
            candidate_actions=candidate_actions,
            actions=actions,
            context=context,
            context_id=context_id,
            latent=latent,
            latent_id=latent_id,
        )

        return actions, prob, log_prob

    def calc_prob_given_actions(  # pyre-ignore
        self,
        actions: torch.Tensor,
        candidate_actions: torch.Tensor,
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
            The candidate actions to calculate the probability.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to calculate the probability of the given actions. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to calculate the probability of the given actions. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to calculate the probability of the given actions. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to calculate the probability of the given actions. Either latent or latent_id must be provided.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (differential).

        """
        early_stage_prob, early_stage_log_prob = (
            self.early_stage_policy.calc_prob_given_actions(
                actions=candidate_actions,
                context=context,
                context_id=context_id,
                is_deterministic=self.is_deterministic_early_stage,
            )
        )
        late_stage_prob, late_stage_log_prob = (
            self.late_stage_policy.calc_prob_given_actions(
                actions=actions,
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                is_deterministic=self.is_deterministic_late_stage,
            )
        )

        log_prob = early_stage_log_prob + late_stage_log_prob
        prob = early_stage_prob * late_stage_prob  # non-differential
        return prob, log_prob


@dataclass
class BaselineSingleStagePolicy(BaseSingleStagePolicy):
    """Single stage ranking/recommendation policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        logits: torch.Tensor,
        n_output_action: int = 1,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_samples, n_action)
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
        _, actions = torch.topk(logits, k=n_output_action, dim=1)
        return actions

    def _log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Input
        ------
        logits: torch.Tensor, shape (n_samples, n_action)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_output_action)
            The index of the actions within the candidate actions.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_sample, n_output_action = actions.shape

        mask = torch.ones_like(logits)
        idxes = torch.arange(n_sample, device=actions.device)
        log_prob = torch.gather(logits, 1, actions).sum(dim=-1)  # (n_samples, )

        for k in range(n_output_action):
            remaining_logits = torch.masked_select(logits, mask.bool()).view(
                n_sample, -1
            )  # (n_samples, n_action - k)
            log_prob = log_prob - torch.logsumexp(remaining_logits, dim=1)

            # .clone() is needed for the gradient to propagate (mask.scatter_ did not work)
            mask = mask.clone()
            mask[idxes, actions[:, k]] = 0

        ### Alternative implementation
        # For each position, the log probability is given by:
        #     log_prob = logit(kth selected) - A - log1p(-exp(B-A))

        # # where A = logsumexp( logit(all) ) and B = logsumexp( logit(selected by kth) )

        # # first calculate the logits(kth selected) (n_samples, n_candidate_action)
        # sample_indices = torch.arange(n_sample, device=self.device)
        # selected_logits = logits[sample_indices, actions]
        # logits_selected_sum = torch.sum(selected_logits, dim=1)  # (n_samples, )

        # # next, we calculate A = logsumexp( logit(all) )
        # logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_samples, )
        # logsumexp_all_sum = logsumexp_all * n_output_action

        # # then, we calculate B = logsumexp( logit(selected by kth) )
        # selected_mask = torch.tril(
        #     torch.ones((n_output_action, n_output_action), device=self.device)
        # )

        # logits_topk = torch.gather(logits, 1, actions)  # (n_samples, n_output_action)
        # exp_logits_topk = torch.exp(logits_topk)

        # logsumexp_selected = torch.log(
        #     torch.sum(exp_logits_topk.unsqueeze(0) * selected_mask.unsqueeze(1), dim=2)
        # ).T  # (n_samples, n_output_action)

        # # finally, we calculate log1p(-exp(B-A))
        # # Note that we need to exclude the first position as no action is selected at the beggining.
        # log1p_selected = torch.log1p(
        #     -torch.exp(logsumexp_selected[:, :-1] - logsumexp_all.unsqueeze(1))
        # )  # (n_samples, n_output_action - 1)
        # log1p_selected_sum = torch.sum(log1p_selected, dim=1)  # (n_samples, )

        # log_prob = logits_selected_sum - logsumexp_all_sum - log1p_selected_sum

        return log_prob

    def retrieve_topk(
        self,
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
        context: Tensor, shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
            )

        topk = self._topk(
            logits=logits,
            n_output_action=n_output_action,
        )
        return topk

    def sample(
        self,
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
        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
            )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        topk = self._topk(
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk

    def sample_actions_with_prob(
        self,
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
        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

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
        logits = self.base_model(
            context=context,
            context_id=context_id,
            latent=latent,
            latent_id=latent_id,
        )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(  # pyre-ignore
                self.device
            )

        actions = self._topk(
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )

        if is_deterministic:
            prob = torch.ones((actions.shape[0],), device=self.device)
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)
        else:
            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits,
                actions=actions,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return actions, prob, log_prob

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
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
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

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
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=actions.shape[1],
            )
            prob = torch.all(actions == selected_actions, dim=1).float()
            log_prob = torch.log(prob + 1e-10)  # avoid log(0)

        else:
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
            )
            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits,
                actions=actions,
            )
            prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob
