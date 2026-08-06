from zeromem.stages import total_memory


def min_zero_stage(num_params: int, world_size: int, budget_bytes: int, activation_memory: int = 0) -> int:
    for stage in (1, 2, 3):
        if total_memory(num_params, world_size, stage, activation_memory) <= budget_bytes:
            return stage
    return 0
