import numpy as np

def optimize_rounding(weights, grid):
    shape = weights.shape
    flat = weights.flatten()
    idx = np.abs(flat[:, None] - grid[None, :]).argmin(axis=1)
    quantized = grid[idx].reshape(shape)
    return quantized
