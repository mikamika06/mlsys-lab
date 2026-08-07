def estimate_traffic(batch_size, num_heads, seq_len, head_dim, dtype_bytes=2, mode="math"):
    qkv = 3 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    output = batch_size * num_heads * seq_len * head_dim * dtype_bytes
    if mode == "math":
        attn_weights_rw = 2 * batch_size * num_heads * seq_len * seq_len * dtype_bytes
        return qkv + output + attn_weights_rw
    else:
        return qkv + output
