import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_quantize_indices(w, block_size=64):
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)
    scales = np.max(np.abs(wb), axis=1)
    scales = np.where(scales == 0, 1.0, scales)
    normalized = wb / scales[:, None]
    diffs = np.abs(normalized[:, :, None] - NF4_LEVELS[None, None, :])
    idx = np.argmin(diffs, axis=-1)
    return idx.reshape(n).astype(np.int64)
