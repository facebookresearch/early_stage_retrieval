# pyre-unsafe
"""Class for learning the joint ranking-wise action embedding."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from early_stage_retrieval.synthetic.dataset import (
    LoggedDataset,
)
from torch import nn
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .base import BaseModelLearner


@dataclass
class JointActionEmbeddingLearner(BaseModelLearner):
    """Training procedure of the joint action embedding model.

    Input
    ------
    model: nn.Module
        The model to train.

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

    def _VAELoss(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kld_weight: float = 0.5,
    ):
        """VAE loss function (for continuous input).

        Input
        ------
        original: torch.Tensor, shape (n_samples, n_output_action * dim_action_emb)
            The original action embedding.

        reconstruction: torch.Tensor, shape (n_samples, n_output_action * dim_action_emb)
            The reconstructed action embedding.

        mu: torch.Tensor, shape (n_samples, dim_joint_action_emb)
            The mean of the latent distribution.

        log_var: torch.Tensor, shape (n_samples, dim_joint_action_emb)
            The log variance of the latent distribution.

        kld_weight: float, default=0.5
            The weight of the KL Divergence loss.

        """
        mse = self._MSELoss(reconstruction, original)
        kld = -kld_weight * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return mse + kld

    def _MSELoss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """MSE loss function (for discrete input).

        Input
        ------
        pred: torch.Tensor, shape (n_samples, n_output_action) or (n_samples, n_output_action * dim_action_emb)
            The predicted action probability.

        target: torch.Tensor, shape (n_samples, n_output_action) or (n_samples, n_output_action * dim_action_emb)
            The target action probability.

        """
        mse = ((pred - target) ** 2).mean()  # pyre-ignore
        return mse

    def train_model_offline(
        self,
        dataset: LoggedDataset,
        kld_weight: float = 0.5,
        pred_weight: float = 0.5,
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

        kld_weight: float, default=0.5
            The weight of the KL Divergence loss.

        pred_weight: float, default=0.5
            The weight of the prediction loss.

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
        train_vae_losses = torch.zeros((n_epoch + 1,), device=self.device)
        val_vae_losses = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )
        train_reg_losses = torch.zeros((n_epoch + 1,), device=self.device)
        val_reg_losses = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )

        vae_loss_fn = self._VAELoss
        reg_loss_fn = self._MSELoss

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[Train action prob regression model: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "train_vae_loss": f"{train_vae_losses[i]:.4g}",
                        "train_reg_loss": f"{train_reg_losses[i]:.4g}",
                        "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
                        "val_vae_loss": f"{val_vae_losses[i // n_epochs_per_log]:.4g}",
                        "val_reg_loss": f"{val_reg_losses[i // n_epochs_per_log]:.4g}",
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
                    reward_ = batch_["reward"].to(self.device, non_blocking=True)

                    recon_, original_, preds_, mu_, log_var_ = model(
                        context=context_,
                        context_id=context_id_,
                        latent_=latent_,
                        latent_id=latent_id_,
                        action_ids=action_ids_,
                    )

                    vae_loss_ = vae_loss_fn(
                        original=original_,
                        reconstruction=recon_,
                        mu=mu_,
                        log_var=log_var_,
                        kld_weight=kld_weight,
                    )
                    reg_loss_ = reg_loss_fn(
                        pred=preds_,
                        target=reward_,
                    )
                    loss_ = vae_loss_ + pred_weight * reg_loss_

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_ / n_steps_per_epoch
                    train_vae_losses[i + 1] += vae_loss_ / n_steps_per_epoch
                    train_reg_losses[i + 1] += reg_loss_ / n_steps_per_epoch

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
                            reward_ = batch_["reward"].to(
                                self.device, non_blocking=True
                            )

                            recon_, original_, preds_, mu_, log_var_ = model(
                                context=context_,
                                context_id=context_id_,
                                latent_=latent_,
                                latent_id=latent_id_,
                                action_ids=action_ids_,
                            )

                            vae_loss_ = vae_loss_fn(
                                original=original_,
                                reconstruction=recon_,
                                mu=mu_,
                                log_var=log_var_,
                                kld_weight=kld_weight,
                            )
                            reg_loss_ = reg_loss_fn(
                                pred=preds_,
                                target=reward_,
                            )
                            val_loss_ = vae_loss_ + pred_weight * reg_loss_

                            val_losses[(i + 1) // n_epochs_per_log] += (
                                val_loss_ * len(batch_["context_id"]) / len(val_dataset)
                            )
                            val_vae_losses[(i + 1) // n_epochs_per_log] += (
                                vae_loss_ * len(batch_["context_id"]) / len(val_dataset)
                            )
                            val_reg_losses[(i + 1) // n_epochs_per_log] += (
                                reg_loss_ * len(batch_["context_id"]) / len(val_dataset)
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
                    "train_vae_losses": train_vae_losses,
                    "train_reg_losses": train_reg_losses,
                    "val_losses": val_losses,
                    "val_vae_losses": val_vae_losses,
                    "val_reg_losses": val_reg_losses,
                },
            )
        else:
            output = model

        return output
