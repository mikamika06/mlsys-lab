import numpy as np


def attention_ai(seqlen: int, dim: int) -> np.ndarray:
    n = float(seqlen)
    d = float(dim)

    qk = (2.0 * n * n * d) / ((2.0 * n * d + n * n) * 4.0)
    softmax = (5.0 * n * n) / ((2.0 * n * n) * 4.0)
    pv = (2.0 * n * n * d) / ((n * n + n * d) * 4.0)

    return np.asarray([qk, softmax, pv], dtype=np.float64)
