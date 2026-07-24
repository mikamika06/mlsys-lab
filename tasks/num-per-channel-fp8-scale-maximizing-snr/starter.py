import numpy as np


def _levels():
    vals = [0.0]
    for e in range(-6, 8):
        for m in range(8):
            vals.append((2.0 ** e) * (1.0 + m / 8.0))
    return np.array(sorted(set(vals)), dtype=np.float64)


_LEVELS = _levels()


def _q_e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    idx = np.searchsorted(_LEVELS, ax)
    idx = np.clip(idx, 1, len(_LEVELS) - 1)
    left = _LEVELS[idx - 1]
    right = _LEVELS[idx]
    chosen = np.where((ax - left) > (right - ax), right, left)
    return sign * np.minimum(chosen, _LEVELS[-1])


def fp8_channel_quantize(W):
    # TODO: This uses one scale for the entire matrix instead of optimizing
    # each channel independently. Rows with smaller ranges lose precision.
    W = np.asarray(W, dtype=np.float64)
    peak = float(np.max(np.abs(W)))
    scale = peak / _LEVELS[-1]
    return _q_e4m3(W / scale) * scale
