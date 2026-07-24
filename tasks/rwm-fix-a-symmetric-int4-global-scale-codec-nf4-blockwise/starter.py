import numpy as np


def nf4_blockwise_dequant(w: np.ndarray, block_size: int = 64) -> np.ndarray:
    # BUG: uses 16 *uniformly*-spaced levels instead of the NF4 quantile
    # codebook, and one *global* scale for the whole array instead of a
    # per-block absmax (block_size is ignored entirely). This wastes
    # resolution on the tails and lets a single outlier blow up the scale
    # for every other block.
    w = np.asarray(w, dtype=np.float64)
    levels = np.linspace(-1.0, 1.0, 16)
    scale = np.max(np.abs(w))
    scale = 1.0 if scale == 0 else scale

    normalized = w / scale
    diffs = np.abs(normalized[:, None] - levels[None, :])
    idx = np.argmin(diffs, axis=-1)
    return levels[idx] * scale
