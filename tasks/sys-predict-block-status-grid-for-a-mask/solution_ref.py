import numpy as np

def block_status_grid(mask, tile_h, tile_w):
    H, W = mask.shape
    n_h = H // tile_h
    n_w = W // tile_w
    area = tile_h * tile_w
    reshaped = mask.reshape(n_h, tile_h, n_w, tile_w)
    tile_sums = reshaped.sum(axis=(1, 3))
    status = np.full((n_h, n_w), 1, dtype=np.int32)
    status[tile_sums == 0] = 2
    status[tile_sums == area] = 0
    return status
