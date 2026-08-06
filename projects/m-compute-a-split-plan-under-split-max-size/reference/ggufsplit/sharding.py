def compute_split_plan_with_tensors(tensors, max_size, max_tensors):
    shards = []
    current_shard = []
    current_size = 0
    for name, size in tensors:
        exceeds_size = bool(current_shard and (current_size + size > max_size))
        exceeds_count = bool(current_shard and (len(current_shard) >= max_tensors))
        if exceeds_size or exceeds_count:
            shards.append(current_shard)
            current_shard = [(name, size)]
            current_size = size
        else:
            current_shard.append((name, size))
            current_size += size
    if current_shard:
        shards.append(current_shard)
    return shards
