def assign_partitions(param_sizes, world_size):
    n = len(param_sizes)
    buckets = [[] for _ in range(world_size)]
    bucket_loads = [0] * world_size
    indexed_params = sorted(enumerate(param_sizes), key=lambda x: x[1], reverse=True)
    for idx, size in indexed_params:
        min_bucket = min(range(world_size), key=lambda i: bucket_loads[i])
        buckets[min_bucket].append(idx)
        bucket_loads[min_bucket] += size
    return buckets
