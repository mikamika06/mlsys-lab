def peak_memory_delta(num_layers, num_heads, head_dim, max_seq_len, dtype_bytes):
    tensor_elements = 2 * num_layers * 1 * num_heads * max_seq_len * head_dim
    base_bytes = tensor_elements * dtype_bytes
    overhead_factor = 1.15
    return int(base_bytes * overhead_factor)
