def optimizer_state_memory(num_params: int, world_size: int, sharded: bool = True) -> int:
    total = 12 * num_params
    if sharded:
        return total // world_size
    return total
