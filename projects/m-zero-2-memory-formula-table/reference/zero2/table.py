def compute_zero2_memory(num_params, world_size, element_size_bytes=4):
    param_bytes = num_params * element_size_bytes
    grad_bytes = num_params * element_size_bytes
    optimizer_states_bytes = (num_params * 12) / world_size
    partitioned_grad_bytes = grad_bytes / world_size
    partitioned_param_bytes = param_bytes / world_size
    total_peak = partitioned_param_bytes + partitioned_grad_bytes + optimizer_states_bytes + (grad_bytes / world_size)
    return {
        "param_bytes": float(param_bytes),
        "grad_bytes": float(grad_bytes),
        "optimizer_states_bytes": float(optimizer_states_bytes),
        "partitioned_grad_bytes": float(partitioned_grad_bytes),
        "partitioned_param_bytes": float(partitioned_param_bytes),
        "total_peak": float(total_peak)
    }
