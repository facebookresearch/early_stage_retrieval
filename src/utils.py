# pyre-unsafe
import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
