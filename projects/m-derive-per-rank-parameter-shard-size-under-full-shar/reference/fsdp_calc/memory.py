def compute_transient_peak_memory(param_size_bytes, grad_size_bytes, activation_size_bytes):
    sharded_param = param_size_bytes
    all_gather_param = param_size_bytes
    return sharded_param + all_gather_param + activation_size_bytes
