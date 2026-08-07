import torch


def partition_optimizer_states(params, world_size, rank):
    flat_params = [p.detach().clone() for p in params]
    total_numel = sum(p.numel() for p in flat_params)
    chunk_size = (total_numel + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, total_numel)

    assigned_params = []
    current_idx = 0
    for p in flat_params:
        p_numel = p.numel()
        p_end = current_idx + p_numel
        if not (p_end <= start or current_idx >= end):
            assigned_params.append(p)
        current_idx = p_end
    return assigned_params
