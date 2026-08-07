def reconstruct_length(grid_shape, block_size):
    g = grid_shape[0] if isinstance(grid_shape, tuple) else grid_shape
    min_len = (g - 1) * block_size + 1 if g > 0 else 0
    max_len = g * block_size
    return (min_len, max_len)
