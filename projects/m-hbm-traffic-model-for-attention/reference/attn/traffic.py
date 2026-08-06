def compute_attention_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes=2):
    q_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    k_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    v_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    scores_rw = batch_size * num_heads * seq_len * seq_len * dtype_bytes * 2
    out_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    return q_bytes + k_bytes + v_bytes + scores_rw + out_bytes
