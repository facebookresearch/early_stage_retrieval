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
        dim_context_emb = base_context_encoder.weight.shape[1]
        dim_latent_emb = base_latent_encoder.weight.shape[1]
        dim_action_emb = base_action_encoder.weight.shape[1]

        self.base_context_encoder = deepcopy(base_context_encoder)
        self.base_latent_encoder = deepcopy(base_latent_encoder)
        self.base_action_encoder = deepcopy(base_action_encoder)

        self.fc1 = nn.Linear(
            dim_context_emb + dim_latent_emb + dim_action_emb * n_output_action,
            dim_hidden1,
        )
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        actions: torch.Tensor,  # (batch_size, n_output_action)
        **kwargs,
    ):
        n_samples = context_id.shape[0]
        context_id = context_id.to(self.base_context_encoder.weight.device)
        latent_id = latent_id.to(self.base_latent_encoder.weight.device)
        action_ids = actions.to(self.base_action_encoder.weight.device)

        context_emb = self.base_context_encoder(context_id)
        latent_emb = self.base_latent_encoder(latent_id)
        action_embs = self.base_action_encoder(action_ids).view(n_samples, -1)

        inputs = torch.cat([context_emb, latent_emb, action_embs], dim=-1)
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x).squeeze()

        return self.sigmoid(logits)  # (batch_size, )


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
        dim_context_emb = base_context_encoder.weight.shape[1]
        dim_latent_emb = base_latent_encoder.weight.shape[1]
        dim_action_emb = base_action_encoder.weight.shape[1]

        self.base_context_encoder = deepcopy(base_context_encoder)
        self.base_latent_encoder = deepcopy(base_latent_encoder)
        self.base_action_encoder = deepcopy(base_action_encoder)

        self.fc1 = nn.Linear(
            dim_context_emb + dim_latent_emb + dim_action_emb * n_output_action,
            dim_hidden1,
        )
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        actions: torch.Tensor,  # (batch_size, n_output_action)
        **kwargs,
    ):
        n_samples = context_id.shape[0]
        context_id = context_id.to(self.base_context_encoder.weight.device)
        latent_id = latent_id.to(self.base_latent_encoder.weight.device)
        action_ids = actions.to(self.base_action_encoder.weight.device)

        context_emb = self.base_context_encoder(context_id)
        latent_emb = self.base_latent_encoder(latent_id)
        action_embs = self.base_action_encoder(action_ids).view(n_samples, -1)

        inputs = torch.cat([context_emb, latent_emb, action_embs], dim=-1)
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x).squeeze()

        return self.softplus(logits)  # (batch_size, )


class JointActionEmbeddingModel(nn.Module):
    """Joint action embedding model for the ranking recommendation.

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

    dim_joint_emb: int = 10
        The dimension of the joint action embedding.

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
        dim_joint_action_emb: int = 10,
        dim_hidden1: int = 100,
        dim_hidden2: int = 20,
    ):
        super().__init__()
        dim_context_emb = base_context_encoder.weight.shape[1]
        dim_latent_emb = base_latent_encoder.weight.shape[1]
        dim_action_emb = base_action_encoder.weight.shape[1]

        self.base_context_encoder = deepcopy(base_context_encoder)
        self.base_latent_encoder = deepcopy(base_latent_encoder)
        self.base_action_encoder = deepcopy(base_action_encoder)

        self.encoder1 = nn.Linear(n_output_action * dim_action_emb, dim_hidden1)
        self.encoder2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.encoder_mu = nn.Linear(dim_hidden2, dim_joint_action_emb)
        self.encoder_log_var = nn.Linear(dim_hidden2, dim_joint_action_emb)

        self.decoder1 = nn.Linear(dim_joint_action_emb, dim_hidden2)
        self.decoder2 = nn.Linear(dim_hidden2, dim_hidden1)
        self.decoder3 = nn.Linear(dim_hidden1, n_output_action * dim_action_emb)

        self.fc1 = nn.Linear(
            dim_context_emb + dim_latent_emb + dim_joint_action_emb, dim_hidden1
        )
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, n_output_action)

        self.relu = nn.ReLU()

    def encode(
        self,
        action_embs: torch.Tensor,  # (batch_size, n_output_action * dim_action_emb)
        **kwargs,
    ):
        x = self.relu(self.encoder1(action_embs))
        x = self.relu(self.encoder2(x))
        mu = self.encoder_mu(x)
        log_var = self.encoder_log_var(x)
        return mu, log_var  # (batch_size, dim_joint_action_emb)

    def reparametrize(
        self,
        mu: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        log_var: torch.Tensor,  # (batch_size, dim_joint_action_emb)
    ):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        action_latent: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        **kwargs,
    ):
        x = self.relu(self.decoder1(action_latent))
        x = self.relu(self.decoder2(x))
        x = self.decoder3(x)
        return x  # (batch_size, n_output_action * dim_action_emb)

    def predict(
        self,
        context_emb: torch.Tensor,  # (batch_size, dim_context_emb)
        latent_emb: torch.Tensor,  # (batch_size, dim_latent_emb)
        action_latent: torch.Tensor,  # (batch_size, dim_joint_action_emb)
    ):
        x = torch.cat([context_emb, latent_emb, action_latent], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # (batch_size, 1)

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        action_ids: torch.Tensor,  # (batch_size, ) or (batch_size, n_output_action)
        **kwargs,
    ):
        n_samples = action_ids.shape[0]
        context_id = context_id.to(self.base_context_encoder.weight.device)
        latent_id = latent_id.to(self.base_latent_encoder.weight.device)
        action_ids = action_ids.to(self.base_action_encoder.weight.device)

        context_emb = self.base_context_encoder(context_id)
        latent_emb = self.base_latent_encoder(latent_id)
        action_embs = self.base_action_encoder(action_ids)

        if action_embs.ndim == 3:
            action_embs = action_embs.view(n_samples, self.encoder1.in_features)

        mu, log_var = self.encode(action_embs)  # (batch_size, dim_joint_action_emb)
        action_latent = self.reparametrize(
            mu, log_var
        )  # (batch_size, dim_joint_action_emb)
        reconstruction = self.decode(
            action_latent
        )  # (batch_size, n_output_action * dim_action_emb)
        prediction = self.predict(
            context_emb=context_emb,
            latent_emb=latent_emb,
            action_latent=action_latent,
        )  # (batch_size, n_output_action)

        return reconstruction, action_embs, prediction, mu, log_var

    def retrieve(
        self,
        action_ids: torch.Tensor,  # (batch_size, ) or (batch_size, n_output_action)
        is_raw_embedding: bool = False,
        **kwargs,
    ):
        n_samples = action_ids.shape[0]
        action_ids = action_ids.to(self.base_action_encoder.weight.device)

        with torch.no_grad():
            action_embs = self.base_action_encoder(action_ids)

        if action_embs.ndim == 3:
            action_embs = action_embs.view(
                n_samples, self.encoder1.in_features
            )  # (batch_size, n_output_action * dim_action_emb)

        if not is_raw_embedding:
            action_embs, _ = self.encode(
                action_embs
            )  # (batch_size, dim_joint_action_emb)

        return action_embs


class EuclideanDistanceModel(nn.Module):
    """Gaussian kernel function."""

    def __init__(
        self,
        scaler: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.scaler = scaler

    def forward(
        self,
        action: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        augmented_action: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        **kwargs,
    ):
        # currently, we support only a simple L2 distance measure
        # however, one can learn context-dependent distance measure as well
        distance = torch.norm(action - augmented_action, dim=-1)
        return distance * self.scaler  # (batch_size, )


class EarthMoverDistanceModel(nn.Module):
    """Earth Mover distance (EMD) kernel function."""

    def __init__(
        self,
        weight_matrix: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        scaler: float = 1.0,
    ):
        super().__init__()
        self.scaler = scaler
        self.weight_matrix = weight_matrix
        self.n_output_action = n_output_action

        if self.weight_matrix is None:
            indices = torch.arange(self.n_output_action).unsqueeze(0)
            abs_diff = torch.abs(indices - indices.T) + 1
            denominator = (
                (self.n_output_action - 1) * (3 * 2 ** (self.n_output_action - 2) - 1)
            ) / 2 ** (self.n_output_action - 1)
            self.weight_matrix = (1 / (2**abs_diff)) / denominator  # pyre-ignore

        else:
            assert (
                self.weight_matrix.shape[0]
                == self.weight_matrix.shape[1]  # pyre-ignore
                == n_output_action
            )

    def forward(
        self,
        action: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        augmented_action: torch.Tensor,  # (batch_size, dim_joint_action_emb)
        **kwargs,
    ):
        batch_size = action.shape[0]
        action = action.view(batch_size, self.n_output_action, -1)
        augmented_action = augmented_action.view(batch_size, self.n_output_action, -1)

        action = action.unsqueeze(1)
        augmented_action = augmented_action.unsqueeze(2)
        distance = ((action - augmented_action) ** 2).sum(-1).sqrt()  # pyre-ignore
        weight_matrix = self.weight_matrix.unsqueeze(0).to(action.device)  # pyre-ignore

        weighted_distance = (distance * weight_matrix).sum(-1).sum(-1)
        return weighted_distance * self.scaler  # (batch_size, )
