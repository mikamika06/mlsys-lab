def predict_max_batch(gpu_memory_bytes, seq_len, num_heads, head_dim):
    bytes_per_elem = 2
    attn_matrix_bytes = seq_len * seq_len * bytes_per_elem
    activations_per_layer = num_heads * head_dim * seq_len * bytes_per_elem * 4
    total_per_batch = attn_matrix_bytes + activations_per_layer
    if total_per_batch <= 0:
        return 1
    safe_memory = gpu_memory_bytes * 0.8
    max_b = int(safe_memory // total_per_batch)
    return max(1, max_b)
