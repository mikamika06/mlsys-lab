import numpy as np
from typing import List, Tuple
from fp4quant.scale import quantize_e8m0


def sweep_block_size(x: np.ndarray, block_sizes: List[int]) -> Tuple[int, float]:
    x_flat = x.astype(np.float64).ravel()
    n = x_flat.size
    best_idx = -1
    best_err = float("inf")
    for idx, b in enumerate(block_sizes):
        if n % b != 0:
            continue
        blocks = x_flat.reshape(-1, b)
        max_vals = np.max(np.abs(blocks), axis=1)
        e8m0 = quantize_e8m0(max_vals)
        scales = np.power(2.0, e8m0.astype(np.float64) - 127.0)
        scales_expanded = np.repeat(scales, b)
        q = np.round(x_flat / np.maximum(scales_expanded, 1e-30))
        q = np.clip(q, -7, 7)
        recon = q * scales_expanded
        err = float(np.mean((x_flat - recon) ** 2))
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx, best_err
