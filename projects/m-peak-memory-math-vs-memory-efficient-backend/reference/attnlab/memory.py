def estimate_peak_memory(batch_size, num_heads, seq_len, head_dim, dtype_bytes=2, backend="math"):
    qkv = 3 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    output = batch_size * num_heads * seq_len * head_dim * dtype_bytes
    if backend == "math":
        attn_weights = batch_size * num_heads * seq_len * seq_len * dtype_bytes
        return qkv + attn_weights + output
    else:
        block_scratch = batch_size * num_heads * seq_len * 64 * dtype_bytes
        return qkv + output + block_scratch
