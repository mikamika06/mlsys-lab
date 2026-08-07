import torch
from torch.utils.checkpoint import checkpoint


def verify_gradient_exactness(model, inputs, strategy):
    def run_pass(use_chk):
        x = inputs.clone().detach().requires_grad_(True)
        out = x
        for i, layer in enumerate(model):
            chk = use_chk and (i < len(strategy) and strategy[i])
            if chk:
                out = checkpoint(layer, out, use_reentrant=False)
            else:
                out = layer(out)
        loss = out.sum()
        loss.backward()
        return x.grad.clone()

    g_base = run_pass(False)
    g_chk = run_pass(True)
    diff = torch.max(torch.abs(g_base - g_chk)).item()
    denom = torch.max(torch.abs(g_base)).item() + 1e-12
    rel_err = diff / denom
    return float(rel_err)
