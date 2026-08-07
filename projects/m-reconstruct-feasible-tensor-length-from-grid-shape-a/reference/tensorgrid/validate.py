import math
from tensorgrid.reconstruct import reconstruct_length


def is_feasible(tensor_len, grid_shape, block_size):
    min_len, max_len = reconstruct_length(grid_shape, block_size)
    return min_len <= tensor_len <= max_len


def find_optimal_grid(tensor_len, block_size):
    return math.ceil(tensor_len / block_size)
