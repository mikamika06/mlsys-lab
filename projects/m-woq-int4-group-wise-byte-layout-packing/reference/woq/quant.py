import numpy as np
from woq.packing import pack_int4_groups, unpack_int4_groups


def compute_quant_error(weights: np.ndarray, group_size: int, smoothed: bool) -> float:
    w = weights.copy()
    if smoothed:
        outlier_factors = np.ones(w.shape[1], dtype=np.float32)
        outlier_factors[::4] = 2.5
        w = w / outlier_factors
    packed, scales = pack_int4_groups(w, group_size)
    reconstructed = unpack_int4_groups(packed, scales, group_size, w.shape)
    if smoothed:
        reconstructed = reconstructed * outlier_factors
    diff = weights - reconstructed
    return float(np.mean(diff ** 2))
