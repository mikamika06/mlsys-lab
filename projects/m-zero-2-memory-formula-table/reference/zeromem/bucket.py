import math

def compute_bucket_count(total_elements, allgather_bucket_size_bytes, dtype_bytes):
    elements_per_bucket = allgather_bucket_size_bytes // dtype_bytes
    if elements_per_bucket <= 0:
        elements_per_bucket = 1
    return math.ceil(total_elements / elements_per_bucket)
