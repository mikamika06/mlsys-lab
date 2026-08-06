def compute_split_plan(tensors, max_size):
    shards = []
    current_shard = []
    current_size = 0
    for name, size in tensors:
        if current_shard and (current_size + size > max_size):
            shards.append(current_shard)
            current_shard = [(name, size)]
            current_size = size
        else:
            current_shard.append((name, size))
            current_size += size
    if current_shard:
        shards.append(current_shard)
    return shards
