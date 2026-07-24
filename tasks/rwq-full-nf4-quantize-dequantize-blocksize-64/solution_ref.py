import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_quantize_dequantize(w: np.ndarray, block_size: int = 64):
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)

    absmax = np.max(np.abs(wb), axis=1)
    absmax_safe = np.where(absmax == 0, 1.0, absmax)
    normalized = wb / absmax_safe[:, None]

    diffs = np.abs(normalized[:, :, None] - NF4_LEVELS[None, None, :])
    idx = np.argmin(diffs, axis=-1).astype(np.uint8)

    deq = (NF4_LEVELS[idx] * absmax_safe[:, None]).reshape(n)
    return idx.reshape(n), deq
