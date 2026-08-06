import torch
import torch.nn as nn
from typing import Dict, Any


class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ToyModelWithAdapter(nn.Module):
    def __init__(self, num_layers: int, dim: int, adapter_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(dim) for _ in range(num_layers)])
        self.adapter = nn.Linear(dim, adapter_dim)
        self.head = nn.Linear(adapter_dim, dim)
        self.checkpointing = False

    def gradient_checkpointing_enable(self):
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if self.checkpointing and self.training:
                def custom_forward(m):
                    def inner(*inputs):
                        return m(*inputs)
                    return inner
                h = torch.utils.checkpoint.checkpoint(custom_forward(layer), h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.adapter(h)
        return self.head(h)


def measure_peak_memory(
    model: nn.Module,
    x: torch.Tensor,
    use_checkpointing: bool
) -> Dict[str, Any]:
    raise NotImplementedError
