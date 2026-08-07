import numpy as np

def calc_bucket_count(total_elements, element_size, allgather_bucket_size_bytes):
    if allgather_bucket_size_bytes <= 0:
        return 1
    bucket_elements = max(1, allgather_bucket_size_bytes // element_size)
    return int(np.ceil(total_elements / bucket_elements))

def toy_reduce_scatter(grads_list, world_size):
    stacked = np.stack(grads_list, axis=0)
    reduced = np.sum(stacked, axis=0)
    chunks = np.array_split(reduced, world_size)
    return [chunks[i].copy() for i in range(world_size)]
