import torch


def compute_unscaled_gradient(param, scale):
    if param.grad is None:
        return None
    return param.grad.detach().clone() / scale


def verify_scaler_unscale(optimizer, scaler):
    scaler.unscale_(optimizer)
    results = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                results[id(p)] = p.grad.detach().clone()
    return results
