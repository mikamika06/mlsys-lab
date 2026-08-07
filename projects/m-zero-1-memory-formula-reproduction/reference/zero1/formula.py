def compute_zero1_memory(num_params, world_size, bytes_per_elem=4, optimizer_type="adam"):
    if optimizer_type == "adam":
        optimizer_bytes_per_param = 12
    else:
        optimizer_bytes_per_param = 4

    model_states = num_params * bytes_per_elem
    optimizer_states = (num_params * optimizer_bytes_per_param) / world_size
    total_memory = model_states + optimizer_states
    return {
        "model_states_bytes": model_states,
        "optimizer_states_bytes": optimizer_states,
        "total_bytes": total_memory
    }
