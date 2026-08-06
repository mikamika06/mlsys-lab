def total_memory(num_params: int, world_size: int, stage: int, activation_memory: int = 0) -> int:
    param_mem = 2 * num_params
    grad_mem = 2 * num_params
    opt_mem = 12 * num_params
    if stage == 1:
        return param_mem + grad_mem + (opt_mem // world_size) + activation_memory
    elif stage == 2:
        return param_mem + (grad_mem // world_size) + (opt_mem // world_size) + activation_memory
    elif stage == 3:
        return (param_mem // world_size) + (grad_mem // world_size) + (opt_mem // world_size) + activation_memory
    raise ValueError("invalid stage")
