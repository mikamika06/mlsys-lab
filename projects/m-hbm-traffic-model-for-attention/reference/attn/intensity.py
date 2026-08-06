from attn.traffic import compute_attention_hbm_bytes


def compute_arithmetic_intensity(batch_size, seq_len, num_heads, head_dim, dtype_bytes=2):
    flops = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    bytes_io = compute_attention_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes)
    return float(flops) / float(bytes_io)
