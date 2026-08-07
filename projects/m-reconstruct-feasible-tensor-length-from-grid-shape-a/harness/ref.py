TEST_CASES = [
    ((1,), 64),
    ((4,), 128),
    ((16,), 256),
    ((0,), 32),
]


def reconstruct_length(grid_shape, block_size):
    g = grid_shape[0] if isinstance(grid_shape, tuple) else grid_shape
    min_len = (g - 1) * block_size + 1 if g > 0 else 0
    max_len = g * block_size
    return (min_len, max_len)


def is_feasible(tensor_len, grid_shape, block_size):
    min_len, max_len = reconstruct_length(grid_shape, block_size)
    return min_len <= tensor_len <= max_len


def find_optimal_grid(tensor_len, block_size):
    import math
    return math.ceil(tensor_len / block_size)
