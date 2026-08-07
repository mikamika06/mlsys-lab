import torch


def measure_optimizer_bytes(model, optimizer_cls, **kwargs):
    optimizer = optimizer_cls(model.parameters(), **kwargs)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    total_bytes = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        total_bytes += v.nelement() * v.element_size()
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, torch.Tensor):
                                total_bytes += sub_v.nelement() * sub_v.element_size()
    return total_bytes
