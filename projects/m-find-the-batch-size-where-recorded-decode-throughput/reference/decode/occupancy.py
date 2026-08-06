def estimate_gemv_occupancy(batch_size, num_heads, head_dim):
    active_threads = batch_size * num_heads * head_dim
    max_threads_per_sm = 2048
    sm_count = 108
    raw_occ = min(1.0, active_threads / (sm_count * max_threads_per_sm))
    return max(0.1, raw_occ)
