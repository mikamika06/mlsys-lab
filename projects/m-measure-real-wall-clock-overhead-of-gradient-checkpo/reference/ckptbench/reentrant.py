import torch
from torch.utils.checkpoint import checkpoint


class InPlaceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        h = self.linear(x)
        h.add_(1.0)
        return h


def test_reentrant_inplace(module_class, x):
    mod = module_class()

    x1 = x.detach().clone().requires_grad_(True)
    try:
        out1 = checkpoint(mod, x1, use_reentrant=True)
        loss1 = out1.sum()
        loss1.backward()
        reentrant_ok = True
    except RuntimeError:
        reentrant_ok = False

    mod_non = module_class()
    x2 = x.detach().clone().requires_grad_(True)
    try:
        out2 = checkpoint(mod_non, x2, use_reentrant=False)
        loss2 = out2.sum()
        loss2.backward()
        non_reentrant_ok = True
    except RuntimeError:
        non_reentrant_ok = False

    return {"reentrant_ok": reentrant_ok, "non_reentrant_ok": non_reentrant_ok}
