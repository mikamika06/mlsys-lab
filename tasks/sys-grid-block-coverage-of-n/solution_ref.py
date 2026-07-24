import numpy as np

def block_coverage(n, block_size):
    grid = (n + block_size - 1) // block_size
    mask = np.zeros((grid, block_size), dtype=bool)
    idx = np.arange(n)
    rows = idx // block_size
    cols = idx % block_size
    mask[rows, cols] = True
    return grid, mask
