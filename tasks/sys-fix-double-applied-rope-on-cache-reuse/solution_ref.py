import numpy as np


def rope(x, positions, base: float = 10000.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    d = x.shape[-1]
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)
    angles = positions[:, None] * theta[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def decode_step(cache, k_raw, v, q_raw, pos):
    d = np.asarray(k_raw).shape[-1]

    # RoPE applied ONCE, at insertion time, to the new token only.
    k = rope(k_raw, pos)[0]
    q = rope(q_raw, pos)[0]

    # Cached keys are already rotated -- reused as-is, never re-rotated.
    cache["k"] = np.vstack([cache["k"], k[None, :]])
    cache["v"] = np.vstack([cache["v"], np.asarray(v, dtype=np.float64)[None, :]])

    scores = (cache["k"] @ q) / np.sqrt(d)
    w = _softmax(scores)
    out = w @ cache["v"]
    return out, cache
