def flatten_pad_shard(params, world_size):
    flat = []
    for p in params:
        stack = [p]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                for sub in reversed(item):
                    stack.append(sub)
            else:
                flat.append(float(item))

    total = len(flat)
    remainder = total % world_size
    pad = 0 if remainder == 0 else (world_size - remainder)
    if pad:
        flat.extend([0.0] * pad)

    shard_size = len(flat) // world_size
    return [flat[i * shard_size:(i + 1) * shard_size] for i in range(world_size)]
