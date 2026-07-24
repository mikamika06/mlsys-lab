import numpy as np

def peak_memory_per_rank(layer_sizes, dtype='float32', cpu_offload=False, num_ranks=1):
    elem_size = np.dtype(dtype).itemsize
    total_bytes = sum(layer_sizes) * elem_size
    largest_bytes = max(layer_sizes) * elem_size if layer_sizes else 0
    if cpu_offload:
        return int(largest_bytes)
    else:
        per_rank_sharded = (3*total_bytes)//num_ranks
        return int(per_rank_sharded + largest_bytes)
