import numpy as np


def compute_bucket_count(param_sizes, allgather_bucket_size):
    if not param_sizes:
        return 0
    buckets = 0
    current_size = 0
    for size in param_sizes:
        if current_size + size > allgather_bucket_size and current_size > 0:
            buckets += 1
            current_size = size
        else:
            current_size += size
    if current_size > 0:
        buckets += 1
    return int(buckets)
