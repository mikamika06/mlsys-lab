def optimizer_state_memory(param_bytes, world_size, zero_stage):
    if zero_stage == 0:
        return param_bytes * 12
    elif zero_stage >= 1:
        return (param_bytes * 12) / world_size
    raise ValueError("Invalid zero stage")
