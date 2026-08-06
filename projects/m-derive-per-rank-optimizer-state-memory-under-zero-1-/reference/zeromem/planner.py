from zeromem.stages import per_rank_memory

def min_zero_stage(param_bytes: int, grad_bytes: int, opt_bytes: int, activation_bytes: int, world_size: int, budget_bytes: int) -> int:
    for stage in range(4):
        if per_rank_memory(param_bytes, grad_bytes, opt_bytes, activation_bytes, world_size, stage) <= budget_bytes:
            return stage
    return -1
