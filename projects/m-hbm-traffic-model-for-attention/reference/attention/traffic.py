def compute_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes):
    q_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    k_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    v_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    o_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    s_bytes = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    return q_bytes + k_bytes + v_bytes + o_bytes + s_bytes
