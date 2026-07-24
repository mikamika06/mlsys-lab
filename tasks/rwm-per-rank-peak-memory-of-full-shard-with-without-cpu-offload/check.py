import numpy as np

def _ref(layer_sizes, dtype='float32', cpu_offload=False, num_ranks=1):
    elem_size = np.dtype(dtype).itemsize
    total_bytes = sum(layer_sizes) * elem_size
    largest_bytes = max(layer_sizes) * elem_size if layer_sizes else 0
    if cpu_offload:
        return int(largest_bytes)
    else:
        per_rank_sharded = (3*total_bytes)//num_ranks
        return int(per_rank_sharded + largest_bytes)

def grade(sol, fx):
    cases = [
        ([1000], 'float32', False, 1),
        ([500, 1500], 'float64', False, 2),
        ([200, 300, 400], 'float32', True, 3),
        ([], 'float32', False, 1),
        ([10**6], 'float16', False, 4)
    ]
    ok = 1.0
    for layer_sizes, dtype, cpu_offload, num_ranks in cases:
        try:
            got = sol.peak_memory_per_rank(layer_sizes, dtype=dtype,
                                            cpu_offload=cpu_offload,
                                            num_ranks=num_ranks)
            ref = _ref(layer_sizes, dtype, cpu_offload, num_ranks)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
