import numpy as np

def compute_optimal_reduce_bucket_size(total_params, num_ranks, latency_sec, bandwidth_bytes_per_sec, elem_bytes=2, max_mem_bytes=1073741824):
    """
    Computes closed-form optimal reduce_bucket_size balancing communication overhead and memory constraints.
    """
    total_grad_bytes = total_params * elem_bytes

    if num_ranks <= 1:
        return min(total_params, int(max_mem_bytes // elem_bytes))

    coeff = 2.0 * (num_ranks - 1) / float(num_ranks)

    def comm_time(b_bytes):
        num_buckets = int(np.ceil(total_grad_bytes / b_bytes))
        return num_buckets * (latency_sec + coeff * (b_bytes / bandwidth_bytes_per_sec))

    min_b = 1024
    max_b = min(total_grad_bytes, max_mem_bytes)

    if min_b >= max_b:
        return int(max_b // elem_bytes)

    candidates = np.logspace(np.log10(min_b), np.log10(max_b), num=200)
    best_b = max_b
    best_t = float('inf')

    for b in candidates:
        b_aligned = int(np.floor(b / 512.0) * 512)
        if b_aligned < min_b:
            b_aligned = min_b
        if b_aligned > max_b:
            b_aligned = max_b
        t = comm_time(b_aligned)
        if t < best_t:
            best_t = t
            best_b = b_aligned

    optimal_numel = int(best_b // elem_bytes)
    return optimal_numel
