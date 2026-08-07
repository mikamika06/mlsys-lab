import math


def compute_bucket_count(total_elements, allgather_bucket_size):
    if allgather_bucket_size <= 0:
        return 1
    return math.ceil(total_elements / allgather_bucket_size)
