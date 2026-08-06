def estimate_hessian_memory(hidden_size, num_parameters, dtype_bytes):
    param_mem = num_parameters * dtype_bytes
    hessian_mem = hidden_size * hidden_size * dtype_bytes
    total_bytes = param_mem + hessian_mem
    return total_bytes
