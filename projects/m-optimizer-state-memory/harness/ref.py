import torch
import torch.nn as nn


def get_test_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.Linear(64, 10),
    )


def get_test_inputs():
    torch.manual_seed(42)
    return torch.randn(16, 64)


def ref_estimate_optimizer_state_bytes(params, optimizer_type, initialized=True):
    if not initialized:
        return 0
    opt = optimizer_type.lower()
    total = 0
    for p in params:
        n = p.numel() * p.element_size()
        if opt in ("sgd_momentum", "momentum"):
            total += n
        elif opt in ("adam", "adamw"):
            total += 2 * n
        elif opt == "sgd":
            total += 0
    return total


def ref_calculate_model_optimizer_footprint(params, optimizer_types):
    param_list = list(params)
    pb = sum(p.numel() * p.element_size() for p in param_list)
    gb = pb
    res = {}
    for opt in optimizer_types:
        sb = ref_estimate_optimizer_state_bytes(param_list, opt, True)
        res[opt] = {
            "param_bytes": pb,
            "grad_bytes": gb,
            "state_bytes": sb,
            "total_bytes": pb + gb + sb,
        }
    return res


def ref_profile_zerograd(model, inputs):
    out = model(inputs)
    loss = out.sum()
    loss.backward()

    grads_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    grads_count = sum(1 for p in model.parameters() if p.grad is not None)

    return {
        "retained_grads_bytes": grads_bytes,
        "retained_grads_count": grads_count,
        "zero_fill_bytes": grads_bytes,
        "zero_fill_count": grads_count,
        "none_fill_bytes": 0,
        "none_fill_count": 0,
        "allocated_bytes_saved": grads_bytes,
    }
