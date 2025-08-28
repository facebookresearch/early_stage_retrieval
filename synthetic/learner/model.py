# pyre-unsafe
"""Models used for the policy traning process."""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class ExpertSelectionModel(nn.Module):
    """Model selector for the mixture-of-expert early stage policy.

    Input
    ------
    n_model: int
        The number of models (mixture-of-experts).

    n_lantent: int
        The number of latent variables.

    """

    def __init__(self, n_model: int, n_latent: int):
        super().__init__()
        self.n_model = n_model
        self.weight_encoder = nn.Embedding(n_latent, n_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        latent_id: torch.Tensor,  # (batch_size, )
        **kwargs,
    ):
        weight = self.softmax(self.weight_encoder(latent_id))
        return weight  # (batch_size, n_model)


class ActionProbModel(nn.Module):
    """Action probability regression model for the joint policy.

    Input
    ------
    base_context_encoder: nn.Embedding
        The (pre-trained) context encoder.

    base_latent_encoder: nn.Embedding
        The (pre-trained) latent encoder.

    base_action_encoder: nn.Embedding
        The (pre-trained) action encoder.

    n_output_action: int = 1
        The number of output actions.

    dim_hidden1: int = 100
        The dimension of the first hidden layer.

    dim_hidden2: int = 20
        The dimension of the second hidden layer.

    """

    def __init__(
        self,
        base_context_encoder: nn.Embedding,
        base_latent_encoder: nn.Embedding,
        base_action_encoder: nn.Embedding,
        n_output_action: int = 1,
        dim_hidden1: int = 100,
        dim_hidden2: int = 20,
    ):
        super().__init__()
        self.n_output_action = n_output_action
        dim_context_emb = base_context_encoder.weight.shape[1]
        dim_latent_emb = base_latent_encoder.weight.shape[1]
        dim_action_emb = base_action_encoder.weight.shape[1]

        self.base_context_encoder = deepcopy(base_context_encoder)
        self.base_latent_encoder = deepcopy(base_latent_encoder)
        self.base_action_encoder = deepcopy(base_action_encoder)

        self.fc1 = nn.Linear(
            dim_context_emb + dim_latent_emb + dim_action_emb,
            dim_hidden1,
        )
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, n_output_action)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        actions: torch.Tensor,  # (batch_size, n_output_action)
        **kwargs,
    ):
        context_id = context_id.to(self.base_context_encoder.weight.device)
        latent_id = latent_id.to(self.base_latent_encoder.weight.device)
        action_ids = actions.to(self.base_action_encoder.weight.device)

        context_emb = self.base_context_encoder(context_id)
        latent_emb = self.base_latent_encoder(latent_id)
        action_embs = self.base_action_encoder(action_ids)

        logits = []
        for k in range(self.n_output_action):  # pyre-ignore
            inputs = torch.cat([context_emb, latent_emb, action_embs[:, k]], dim=-1)
            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            logits.append(self.fc3(x)[:, k].unsqueeze(-1))

        logits = torch.cat(logits, dim=-1)
        return self.sigmoid(logits)  # (batch_size, n_output_action)


class KernelDensityModel(nn.Module):
    """Action probability regression model for the joint policy.

    Input
    ------
    base_context_encoder: nn.Embedding
        The (pre-trained) context encoder.

    base_latent_encoder: nn.Embedding
        The (pre-trained) latent encoder.

    base_action_encoder: nn.Embedding
        The (pre-trained) action encoder.

    dim_hidden1: int = 100
        The dimension of the first hidden layer.

    dim_hidden2: int = 20
        The dimension of the second hidden layer.

    """

    def __init__(
        self,
        base_context_encoder: nn.Embedding,
        base_latent_encoder: nn.Embedding,
        base_action_encoder: nn.Embedding,
        n_output_action: int = 1,
        dim_hidden1: int = 100,
        dim_hidden2: int = 20,
    ):
        super().__init__()
        self.n_output_action = n_output_action
        dim_context_emb = base_context_encoder.weight.shape[1]
        dim_latent_emb = base_latent_encoder.weight.shape[1]
        dim_action_emb = base_action_encoder.weight.shape[1]

        self.base_context_encoder = deepcopy(base_context_encoder)
        self.base_latent_encoder = deepcopy(base_latent_encoder)
        self.base_action_encoder = deepcopy(base_action_encoder)

        self.fc1 = nn.Linear(
            dim_context_emb + dim_latent_emb + dim_action_emb,
            dim_hidden1,
        )
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, n_output_action)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        actions: torch.Tensor,  # (batch_size, n_output_action)
        **kwargs,
    ):
        context_id = context_id.to(self.base_context_encoder.weight.device)
        latent_id = latent_id.to(self.base_latent_encoder.weight.device)
        action_ids = actions.to(self.base_action_encoder.weight.device)

        context_emb = self.base_context_encoder(context_id)
        latent_emb = self.base_latent_encoder(latent_id)
        action_embs = self.base_action_encoder(action_ids)

        logits = []
        for k in range(self.n_output_action):
            inputs = torch.cat([context_emb, latent_emb, action_embs[:, k]], dim=-1)
            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            logits.append(self.fc3(x)[:, k].unsqueeze(-1))

        logits = torch.cat(logits, dim=-1)
        return self.softplus(logits)  # (batch_size, n_output_action)
