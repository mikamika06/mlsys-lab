def reduce_scatter_chunks(buffers, world_size):
    """Return reduced shards for each rank after a reduce-scatter operation."""
    chunk_size = len(buffers[0]) // world_size
    result = []

    for owner in range(world_size):
        start = (owner + 1) * chunk_size
        end = start + chunk_size
        reduced = [0] * chunk_size
        for buf in buffers:
            chunk = buf[start:end]
            for i, value in enumerate(chunk):
                reduced[i] += value
        result.append(reduced)

    return result
