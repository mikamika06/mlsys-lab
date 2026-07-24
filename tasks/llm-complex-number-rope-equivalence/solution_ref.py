import numpy as np

def rope_complex(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) via complex multiplication.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    batch, seq_len, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    freqs = 10000 ** (-np.arange(0, dim // 2) / (dim / 2))
    theta = pos[:, None] * freqs[None, :]

    a = x[..., ::2]
    b = x[..., 1::2]
    z = a + 1j * b
    rot = np.exp(1j * theta)
    z_rot = z * rot

    out = np.empty_like(x)
    out[..., ::2] = z_rot.real
    out[..., 1::2] = z_rot.imag
    return out
