import torch


def get_test_cases():
    torch.manual_seed(42)
    cases = []
    for size in [16, 64, 256]:
        p = torch.nn.Parameter(torch.randn(size))
        p.grad = torch.randn(size) * 10.0
        cases.append([p])
    return cases


def estimate_cost(parameters, max_norm):
    total_elements = 0
    num_tensors = 0
    for p in parameters:
        if p.grad is not None:
            total_elements += p.grad.numel()
            num_tensors += 1
    flops = total_elements * 2
    bytes_moved = total_elements * 4 * 2
    estimated_time_ms = (flops / 1e9) + (num_tensors * 0.01)
    return {
        "num_tensors": num_tensors,
        "total_elements": total_elements,
        "estimated_time_ms": float(estimated_time_ms),
        "bytes_moved": float(bytes_moved)
    }


def compute_grad_norm(parameters, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(g.detach().abs().max() for g in grads)
        return total_norm
    else:
        sum_sq = torch.tensor(0.0, device=device)
        for g in grads:
            sum_sq = sum_sq + torch.sum(g.detach().pow(norm_type))
        total_norm = sum_sq.pow(1.0 / norm_type)
        return total_norm


def clip_grads(parameters, max_norm, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = compute_grad_norm(parameters, norm_type=norm_type)
    max_norm = float(max_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if clip_coef_clamped < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef_clamped)

    return total_norm
