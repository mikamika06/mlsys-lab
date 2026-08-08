import math
import numpy as np


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if norm_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return mid


def build_nf4_codebook():
    P = [(i + 0.5) / 15.0 for i in range(8)] + [0.5 + (i + 0.5) / 16.0 for i in range(8)]
    V = np.array([norm_ppf(p) for p in P])
    return V / np.max(np.abs(V))


def quantize_tensor(w, codebook, block_size):
    w_blocks = w.reshape(-1, block_size)
    absmaxes = np.max(np.abs(w_blocks), axis=1)

    safe_absmaxes = np.where(absmaxes == 0, 1.0, absmaxes)
    w_scaled = w_blocks / safe_absmaxes[:, np.newaxis]

    diffs = np.abs(w_scaled[..., np.newaxis] - codebook)
    indices = np.argmin(diffs, axis=-1)

    return indices, absmaxes


def dequantize_tensor(indices, absmaxes, codebook):
    w_scaled = codebook[indices]
    w_blocks = w_scaled * absmaxes[:, np.newaxis]
    return w_blocks.flatten()
