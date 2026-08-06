import math
import numpy as np


def _levels():
    vals = [0.0]
    for e in range(-6, 8):
        for m in range(8):
            vals.append((2.0**e) * (1.0 + m / 8.0))
    return np.array(sorted(set(vals)), dtype=np.float64)


_LEVELS = _levels()


def _q_single(x):
    if x > 0.0:
        s = 1.0
    elif x < 0.0:
        s = -1.0
    else:
        s = 0.0

    ax = abs(x)

    idx = len(_LEVELS)
    for i in range(len(_LEVELS)):
        if _LEVELS[i] >= ax:
            idx = i
            break

    if idx < 1:
        idx = 1
    elif idx >= len(_LEVELS):
        idx = len(_LEVELS) - 1

    left = _LEVELS[idx - 1]
    right = _LEVELS[idx]

    if (ax - left) > (right - ax):
        chosen = right
    else:
        chosen = left

    if chosen > _LEVELS[-1]:
        chosen = _LEVELS[-1]

    return s * chosen


def _q_e4m3(x):
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)
    for idx in np.ndindex(x_arr.shape):
        out[idx] = _q_single(float(x_arr[idx]))
    return out


def fp8_channel_quantize(W):
    W_arr = np.asarray(W, dtype=np.float64)
    out = np.empty_like(W_arr, dtype=np.float64)

    for i in range(W_arr.shape[0]):
        row = W_arr[i]
        peak = 0.0
        for j in range(len(row)):
            v = abs(float(row[j]))
            if v > peak:
                peak = v

        if peak == 0.0:
            for j in range(len(row)):
                out[i, j] = row[j]
            continue

        start = math.log10(peak / _LEVELS[-1])
        stop = math.log10(peak)
        step = (stop - start) / 191.0

        best = np.empty(len(row), dtype=np.float64)
        best_loss = float("inf")

        for k in range(192):
            scale = 10.0 ** (start + k * step)
            cand = np.empty(len(row), dtype=np.float64)
            loss = 0.0
            for j in range(len(row)):
                val = float(row[j])
                c = _q_single(val / scale) * scale
                cand[j] = c
                diff = c - val
                loss += diff * diff

            if loss < best_loss:
                best_loss = loss
                for j in range(len(row)):
                    best[j] = cand[j]

        for j in range(len(row)):
            out[i, j] = best[j]

    return out
