def simulate_traffic(mesh_shape, tensor_sizes, sharding_strategy):
    r, c = mesh_shape
    total_ranks = r * c
    link_loads = [[0.0 for _ in range(c)] for _ in range(r)]
    for size in tensor_sizes:
        if sharding_strategy == "HYBRID_SHARD":
            shard_size = size / total_ranks
            for row in range(r):
                for col in range(c):
                    link_loads[row][col] += shard_size * 1.5
        elif sharding_strategy == "DDP":
            for row in range(r):
                for col in range(c):
                    link_loads[row][col] += size * 2.0
        else:
            shard_size = size / total_ranks
            for row in range(r):
                for col in range(c):
                    link_loads[row][col] += shard_size
    return link_loads
