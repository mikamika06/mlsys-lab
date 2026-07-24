import numpy as np


def yarn_ramp_temperature(
    q,
    k,
    inv_freq,
    positions,
    beta_fast,
    beta_slow,
    scale,
    temperature,
):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)

    dim = np.arange(inv_freq.shape[0], dtype=np.float64)
    ramp = np.clip(
        (dim - beta_slow) / (beta_fast - beta_slow),
        0.0,
        1.0,
    )
    freq = inv_freq / (1.0 + ramp * (scale - 1.0))

    def apply_rope(x):
        angles = np.asarray(positions, dtype=np.float64)[:, None] * freq[None, :]
        c = np.cos(angles)
        s = np.sin(angles)
        out = np.empty_like(x, dtype=np.float64)
        x0 = x[:, 0::2]
        x1 = x[:, 1::2]
        out[:, 0::2] = x0 * c - x1 * s
        out[:, 1::2] = x0 * s + x1 * c
        return out

    qr = apply_rope(q)
    kr = apply_rope(k)
    return (qr @ kr.T) / (np.sqrt(q.shape[1]) * np.sqrt(temperature))
