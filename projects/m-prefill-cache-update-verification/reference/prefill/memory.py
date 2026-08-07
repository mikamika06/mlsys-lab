def peak_memory_delta(num_layers, num_heads, head_dim, max_seq_len, dtype_bytes):
    return float(2 * num_layers * 1 * num_heads * head_dim * max_seq_len * dtype_bytes)
