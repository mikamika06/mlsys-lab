def crossover_point(hidden_size, num_heads, world_size):
    return float(hidden_size * world_size)
