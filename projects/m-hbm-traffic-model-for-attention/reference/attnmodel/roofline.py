from attnmodel.arithmetic import arithmetic_intensity

def classify_bound(batch_size, seq_len, num_heads, head_dim, peak_flops, peak_bw, bytes_per_elem=2):
    intensity = arithmetic_intensity(batch_size, seq_len, num_heads, head_dim, bytes_per_elem)
    ridge = peak_flops / peak_bw
    if intensity < ridge:
        return "memory-bound"
    return "compute-bound"
