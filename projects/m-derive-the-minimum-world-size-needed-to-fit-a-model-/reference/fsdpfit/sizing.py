def calculate_min_world_size(param_bytes, optimizer_bytes_per_param, grad_bytes_per_param, activation_bytes, budget_bytes):
    for ws in range(1, 1024):
        per_rank_params = param_bytes / ws
        per_rank_grads = grad_bytes_per_param * (param_bytes / ws)
        per_rank_opt = optimizer_bytes_per_param * (param_bytes / ws)
        total = per_rank_params + per_rank_grads + per_rank_opt + activation_bytes
        if total <= budget_bytes:
            return ws
    return 1024
