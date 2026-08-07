import numpy as np


def compute_l2_hit_rate(M: int, N: int, K: int, block_m: int, block_n: int, grouped: bool) -> float:
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n
    if not grouped:
        hit_rate = 0.45 + 0.05 * (1.0 / (1.0 + 0.001 * grid_m * grid_n))
    else:
        hit_rate = 0.75 + 0.10 * (1.0 / (1.0 + 0.001 * grid_m * grid_n))
    return float(np.clip(hit_rate, 0.0, 1.0))
