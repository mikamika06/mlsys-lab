import numpy as np


def online_softmax_stream(x: np.ndarray, B: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    m = -np.inf
    s = 0.0
    peak_elements = 2

    for start in range(0, n, B):
        block = x[start:start + B]
        peak_elements = max(peak_elements, block.size + 2)

        block_max = np.max(block)
        new_m = max(m, block_max)
        s = s * np.exp(m - new_m) + np.sum(np.exp(block - new_m))
        m = new_m

    out = np.empty_like(x)
    peak_elements = max(peak_elements, B + 2)

    for start in range(0, n, B):
        block = x[start:start + B]
        out[start:start + B] = np.exp(block - m) / s

    return out, peak_elements
