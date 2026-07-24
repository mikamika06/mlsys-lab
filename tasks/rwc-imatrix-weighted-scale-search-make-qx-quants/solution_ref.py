import numpy as np


def make_qx_quants(x, w, nmax):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(x))
    if amax == 0:
        return -1, np.zeros(x.shape, dtype=np.int64)

    base_scale = amax / nmax
    best_idx = -1
    best_err = None
    best_codes = None
    for k in range(-15, 16):
        idx = k + 15
        scale = base_scale * (1.0 + k / 32.0)
        if scale == 0:
            continue
        codes = np.clip(np.round(x / scale), -nmax, nmax)
        err = float(np.sum(w * (x - scale * codes) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
            best_codes = codes.astype(np.int64)
    return best_idx, best_codes
