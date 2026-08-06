import torch
import torch.nn as nn
import torch.nn.functional as F
from tp.operators import column_parallel_matmul, row_parallel_matmul


class TensorParallelMLP(nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim: int, process_group=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.process_group = process_group

        tp_size = torch.distributed.get_world_size(process_group) if torch.distributed.is_initialized() else 1
        assert ffn_dim % tp_size == 0, "ffn_dim must be divisible by tp_size"

        self.local_ffn_dim = ffn_dim // tp_size
        self.w1 = nn.Parameter(torch.empty(self.local_ffn_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, self.local_ffn_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = column_parallel_matmul(x, self.w1, self.process_group)
        h = F.gelu(h)
        out = row_parallel_matmul(h, self.w2, self.process_group)
        return out
