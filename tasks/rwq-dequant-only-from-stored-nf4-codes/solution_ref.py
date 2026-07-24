import numpy as np

_NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float32)


def nf4_dequantize(idx: np.ndarray, absmax: np.ndarray, block_size: int = 64) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    absmax = np.asarray(absmax, dtype=np.float32)

    n = idx.size
    n_blocks = n // block_size

    codes = _NF4_LEVELS[idx]
    scales = np.repeat(absmax[:n_blocks], block_size)
    return codes * scales
