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
    if use_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    model.train()

    saved_tensors = []

    def pack_hook(t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            saved_tensors.append(t.numel() * t.element_size())
        return t

    def unpack_hook(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        out = model(x)
        loss = out.sum()
        loss.backward()

    activation_bytes = sum(saved_tensors)
    param_grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    peak_bytes = activation_bytes + param_grad_bytes

    return {
        "peak_bytes": peak_bytes,
        "saved_count": len(saved_tensors),
        "use_checkpointing": use_checkpointing
    }
