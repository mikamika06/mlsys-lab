from attention.traffic import compute_hbm_bytes


def compute_arithmetic_intensity(batch_size, seq_len, num_heads, head_dim, dtype_bytes):
    flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim + 2 * batch_size * num_heads * seq_len * seq_len
    bytes_io = compute_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes)
    return flops / bytes_io
