import numpy as np
from typing import List, Tuple


def reference_unpack(packed_bytes: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_bytes, dtype=np.uint8)
    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F
    unpacked = np.empty((packed.size * 2,), dtype=np.uint8)
    unpacked[0::2] = low_nibbles.ravel()
    unpacked[1::2] = high_nibbles.ravel()
    return unpacked.reshape(packed.shape[:-1] + (-1,))


def reference_quantize_e8m0(scale_fp32: np.ndarray) -> np.ndarray:
    scales = np.maximum(scale_fp32, 1e-30)
    log2_val = np.log2(scales)
    floor_exp = np.floor(log2_val)
    frac = log2_val - floor_exp
    exp = np.where(frac > 0.5, floor_exp + 1.0, np.where(frac < 0.5, floor_exp, np.where(np.abs(floor_exp) % 2 == 1.0, floor_exp + 1.0, floor_exp)))
    e8m0_biased = np.clip(exp + 127.0, 0, 255).astype(np.uint8)
    return e8m0_biased


def reference_sweep(x: np.ndarray, block_sizes: List[int]) -> Tuple[int, float]:
    x_flat = x.astype(np.float64).ravel()
    n = x_flat.size
    best_idx = -1
    best_err = float("inf")
    for idx, b in enumerate(block_sizes):
        if n % b != 0:
            continue
        blocks = x_flat.reshape(-1, b)
        max_vals = np.max(np.abs(blocks), axis=1)
        e8m0 = reference_quantize_e8m0(max_vals)
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
