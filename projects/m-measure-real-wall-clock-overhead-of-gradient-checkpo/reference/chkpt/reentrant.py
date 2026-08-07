import torch
from torch.utils.checkpoint import checkpoint

def run_reentrant_test(model, inputs, use_reentrant):
    x = inputs.clone().detach().requires_grad_(True)

    class InPlaceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, val):
            out = self.linear(val)
            out.add_(1.0)
            return out

    mod = InPlaceModule()

    try:
        if use_reentrant:
            out = checkpoint(mod, x, use_reentrant=True)
        else:
            out = checkpoint(mod, x, use_reentrant=False)
        loss = out.sum()
        loss.backward()
        return True, None
    except Exception as e:
        return False, str(e)
