def zero2_memory_breakdown(num_params, world_size, bytes_per_param=2, optimizer_bytes_per_param=12, activation_bytes=0):
    p_mem = num_params * bytes_per_param
    g_mem = (num_params * bytes_per_param) / world_size
    o_mem = (num_params * optimizer_bytes_per_param) / world_size
    a_mem = activation_bytes
    total = p_mem + g_mem + o_mem + a_mem
    return {
        "params": float(p_mem),
        "gradients": float(g_mem),
        "optimizer_states": float(o_mem),
        "activations": float(a_mem),
        "total": float(total)
    }
