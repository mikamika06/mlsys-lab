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
    """One autoregressive decode step for a single attention head.

    cache: {"k": (t, d) already-rotated cached keys, "v": (t, d) cached values}.
    k_raw, q_raw: (d,) raw, un-rotated key/query for the new token.
    v: (d,) value for the new token.
    pos: absolute position (int) of the new token.

    Must apply RoPE to k_raw and q_raw exactly once (at insertion), append
    the rotated key and the value to the cache, and return the causal
    scaled dot-product attention output over everything cached so far.
    Returns (output: (d,) float64 array, updated cache dict).
    """
    d = np.asarray(k_raw).shape[-1]

    k = rope(k_raw, pos)[0]

    # BUG: re-applies RoPE to the entire cache (including already-rotated
    # entries from previous steps) on every call, before appending the new
    # key -- compounding the rotation instead of applying it once at insert.
    if cache["k"].shape[0] > 0:
        cache["k"] = rope(cache["k"], np.arange(cache["k"].shape[0]))

    cache["k"] = np.vstack([cache["k"], k[None, :]])
    cache["v"] = np.vstack([cache["v"], np.asarray(v, dtype=np.float64)[None, :]])

    q = rope(q_raw, pos)[0]
    scores = (cache["k"] @ q) / np.sqrt(d)
    w = _softmax(scores)
    out = w @ cache["v"]
    return out, cache
