import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class BrokenModel(nn.Module):
    def __init__(self, num_layers=8, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(dim) for _ in range(num_layers)])
        self.external_state = torch.zeros(1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    self.external_state += 1.0
                    return module(*inputs)
                return custom_forward
            out = checkpoint(create_custom_forward(layer), out, use_reentrant=False)
        return out

class FixedModel(nn.Module):
    def __init__(self, num_layers=8, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = checkpoint(layer, out, use_reentrant=False)
        return out

def run_reproduction_test():
    model = BrokenModel()
    x = torch.randn(2, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    try:
        loss.backward()
        return False
    except RuntimeError:
        return True

def run_fixed_test():
    model = FixedModel()
    x = torch.randn(2, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    return True
