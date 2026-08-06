from zeromem.optimizer import total_memory

def minimum_zero_stage(param_bytes, world_size, budget_bytes):
    for stage in range(4):
        if total_memory(param_bytes, world_size, stage) <= budget_bytes:
            return stage
    return -1
