def bin_pack_partition(param_sizes, world_size):
    total_size = sum(param_sizes)
    capacity = (total_size + world_size - 1) // world_size
    buckets = [[] for _ in range(world_size)]
    bucket_loads = [0] * world_size
    curr_bucket = 0
    for idx, size in enumerate(param_sizes):
        if curr_bucket < world_size - 1 and bucket_loads[curr_bucket] + size > capacity and bucket_loads[curr_bucket] > 0:
            curr_bucket += 1
        buckets[curr_bucket].append(idx)
        bucket_loads[curr_bucket] += size
    return buckets
