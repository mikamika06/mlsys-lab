def decode_arithmetic_intensity(g, n_q, d, seq_len):
    n_kv = n_q / g
    flops = 4.0 * n_q * seq_len * d
    kv_bytes = 4.0 * seq_len * d * n_kv
    q_bytes = 2.0 * n_q * d
    return flops / (kv_bytes + q_bytes)
