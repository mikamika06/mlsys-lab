CONFIGS = [
    {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64, "dtype_bytes": 2, "peak_flops": 1e12, "peak_bandwidth": 1e9},
    {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 128, "dtype_bytes": 2, "peak_flops": 2e12, "peak_bandwidth": 1.5e9},
    {"batch_size": 4, "seq_len": 2048, "num_heads": 32, "head_dim": 64, "dtype_bytes": 4, "peak_flops": 5e12, "peak_bandwidth": 2e9},
]


def ref_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes):
    q_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    k_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    v_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    o_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    s_bytes = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    return q_bytes + k_bytes + v_bytes + o_bytes + s_bytes


def ref_arithmetic_intensity(batch_size, seq_len, num_heads, head_dim, dtype_bytes):
    flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim + 2 * batch_size * num_heads * seq_len * seq_len
    bytes_io = ref_hbm_bytes(batch_size, seq_len, num_heads, head_dim, dtype_bytes)
    return flops / bytes_io


def ref_classify_roofline(intensity, peak_flops, peak_bandwidth):
    ridge = peak_flops / peak_bandwidth
    if intensity < ridge:
        return "memory_bound"
    return "compute_bound"
