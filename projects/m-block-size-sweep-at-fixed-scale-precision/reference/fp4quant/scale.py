import numpy as np


def quantize_e8m0(scale_fp32: np.ndarray) -> np.ndarray:
    scales = np.maximum(scale_fp32, 1e-30)
    log2_val = np.log2(scales)
    floor_exp = np.floor(log2_val)
    frac = log2_val - floor_exp
    exp = np.where(frac > 0.5, floor_exp + 1.0, np.where(frac < 0.5, floor_exp, np.where(np.abs(floor_exp) % 2 == 1.0, floor_exp + 1.0, floor_exp)))
    e8m0_biased = np.clip(exp + 127.0, 0, 255).astype(np.uint8)
    return e8m0_biased
