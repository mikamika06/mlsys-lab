import numpy as np
from scipy.stats import norm


def get_nf4_codebook() -> np.ndarray:
    q = np.linspace(0, 1, 17)
    quantiles = norm.ppf(0.5 * (q[:-1] + q[1:]))
    quantiles = quantiles / quantiles[-1]

    values = np.zeros(16)
    for i in range(8):
        values[i] = quantiles[i]
    for i in range(8, 16):
        values[i] = quantiles[i + 1]

    values = values / values[-1]
    values = np.clip(values, -1.0, 1.0)
    values[7] = 0.0
    values[8] = 0.0
    values = np.unique(values)
    if len(values) < 16:
        values = np.linspace(-1, 1, 16)
    return values.astype(np.float32)


def get_fp4_codebook() -> np.ndarray:
    vals = np.array([
        0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5,
        0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0
    ], dtype=np.float32)
    vals = vals / vals[-1] * 1.0
    return vals
