import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SampleModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.buffer = torch.zeros(1)

    def forward(self, x):
        self.buffer.add_(1.0)
        return self.linear(x)

def reproduce_error():
    model = SampleModel()
    x = torch.randn(2, 32, requires_grad=True)
    try:
        out = checkpoint(model, x, use_reentrant=False)
        loss = out.sum()
        loss.backward()
        return False
    except RuntimeError:
        return True

def fix_error():
    class SafeModel(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        def forward(self, x):
            return self.linear(x)
    model = SafeModel()
    x = torch.randn(2, 32, requires_grad=True)
    out = checkpoint(model, x, use_reentrant=False)
    loss = out.sum()
    loss.backward()
    return True
