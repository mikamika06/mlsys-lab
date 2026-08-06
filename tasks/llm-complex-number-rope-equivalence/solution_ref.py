import math
import numpy as np

def rope_complex(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) via complex multiplication.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    batch, seq_len, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"

    half_dim = dim // 2
    out = np.empty_like(x)

    freqs = [0.0] * half_dim
    for i in range(half_dim):
        freqs[i] = 10000.0 ** (-(i / (dim / 2)))

    for b in range(batch):
        for s in range(seq_len):
            p = float(pos[s])
            for i in range(half_dim):
                theta = p * freqs[i]
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)

                a = float(x[b, s, 2 * i])
                b_val = float(x[b, s, 2 * i + 1])

                out[b, s, 2 * i] = a * cos_t - b_val * sin_t
                out[b, s, 2 * i + 1] = a * sin_t + b_val * cos_t

    return out
