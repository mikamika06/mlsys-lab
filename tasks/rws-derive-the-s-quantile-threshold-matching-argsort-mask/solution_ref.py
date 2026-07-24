import numpy as np


def quantile_keep_mask(w: np.ndarray, s: float):
    flat = np.asarray(w).reshape(-1)
    mags = np.abs(flat).astype(np.float64)

    threshold = float(np.quantile(mags, s))
    k = int(np.ceil((1.0 - s) * len(mags)))

    indices = np.arange(len(mags))
    order = np.lexsort((indices, -mags))

    mask = np.zeros(len(mags), dtype=bool)
    mask[order[:k]] = True

    return threshold, mask.reshape(np.asarray(w).shape)
