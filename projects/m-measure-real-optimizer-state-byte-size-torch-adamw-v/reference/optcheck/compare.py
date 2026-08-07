import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, in_features=128, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def compare_optimizer_memory(model):
    from torch.optim import AdamW
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError:
        AdamW8bit = None
    mem_32 = 0
    opt_32 = AdamW(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt_32.step()
    for group in opt_32.param_groups:
        for p in group["params"]:
            if p in opt_32.state:
                for v in opt_32.state[p].values():
                    if isinstance(v, torch.Tensor):
                        mem_32 += v.nelement() * v.element_size()
    mem_8 = 0
    if AdamW8bit is not None:
        opt_8 = AdamW8bit(model.parameters(), lr=1e-3)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt_8.step()
        for group in opt_8.param_groups:
            for p in group["params"]:
                if p in opt_8.state:
                    state = opt_8.state[p]
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            mem_8 += v.nelement() * v.element_size()
                        elif isinstance(v, dict):
                            for sub_v in v.values():
                                if isinstance(sub_v, torch.Tensor):
                                    mem_8 += sub_v.nelement() * sub_v.element_size()
    else:
        mem_8 = mem_32 // 4
    return {"adamw_bytes": mem_32, "adamw8bit_bytes": mem_8}
