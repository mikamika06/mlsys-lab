import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_blockwise_dequant(w: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Quantize-then-dequantize `w` through the NF4-blockwise codec: per-block
    absmax scale + nearest-level lookup in the fixed 16-value NF4 codebook.
    """
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size

    scales = []
    for i in range(nb):
        max_val = 0.0
        for j in range(block_size):
            val = abs(w[i * block_size + j])
            if val > max_val:
                max_val = val
        if max_val == 0.0:
            max_val = 1.0
        scales.append(max_val)

    xhat_list = []
    for i in range(nb):
        scale = scales[i]
        for j in range(block_size):
            normalized_val = w[i * block_size + j] / scale
            min_diff = float("inf")
            best_idx = 0
            for k in range(16):
                diff = abs(normalized_val - NF4_LEVELS[k])
                if diff < min_diff:
                    min_diff = diff
                    best_idx = k
            xhat_list.append(NF4_LEVELS[best_idx] * scale)

    return np.array(xhat_list, dtype=np.float64)
