import torch
from torch.utils.checkpoint import checkpoint

class CallCounterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.linear(x)

def count_recomputations(model_fn, inputs):
    mod = CallCounterModule()
    x = inputs.clone().detach().requires_grad_(True)
    out = checkpoint(mod, x, use_reentrant=False)
    loss = out.sum()
    loss.backward()
    return mod.call_count
