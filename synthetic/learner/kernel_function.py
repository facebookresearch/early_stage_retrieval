# pyre-unsafe
"""Implementation of the kernel functions."""

import torch


def gaussian_kernel(
    distance: torch.Tensor,
    kernel_bandwidth: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Compute the gaussian kernel.

    Input
    ------
    distance: torch.Tensor, shape (n_samples, n_output_action)
        The distance between the action and the augmented action.

    kernel_bandwidth: float, default=1.0
        The kernel bandwidth.

    Output
    ------
    kernel_weight: torch.Tensor, shape (n_samples, )
        The kernel weight.

    """
    scaler = 1 / (2 * torch.pi * kernel_bandwidth**2) ** (1 / 2)
    weight = scaler * torch.exp(
        -(distance**2) / (2 * kernel_bandwidth**2)  # pyre-ignore
    )
    return weight
