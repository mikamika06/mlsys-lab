import torch
import torch.nn as nn


class TensorParallelMLP(nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim: int, process_group=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
