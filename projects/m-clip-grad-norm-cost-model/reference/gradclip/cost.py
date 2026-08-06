import torch


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
