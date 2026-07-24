import numpy as np


def _wmse(x, x_hat, w):
    return float(np.sum(w * (x - x_hat) ** 2) / np.sum(w))


def _q4_0(x):
    amax = np.max(np.abs(x))
    d = amax / 8.0 if amax != 0 else 1e-12
    codes = np.clip(np.round(x / d), -8, 7)
    return d * codes


def _search_scale(x, weight):
    amax = np.max(np.abs(x))
    d0 = amax / 8.0 if amax != 0 else 1e-12
    best_err = None
    best_recon = None
    for k in range(-15, 16):
        d = d0 * (1.0 + k / 32.0)
        if d == 0:
            continue
        codes = np.clip(np.round(x / d), -8, 7)
        recon = d * codes
        err = float(np.sum(weight * (x - recon) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_recon = recon
    return best_recon


def compare_q4_variants(x: np.ndarray, w: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    recon_q4_0 = _q4_0(x)
    recon_q4_k = _search_scale(x, np.ones_like(w))
    recon_imatrix = _search_scale(x, w)

    errors = np.array(
        [_wmse(x, recon_q4_0, w), _wmse(x, recon_q4_k, w), _wmse(x, recon_imatrix, w)],
        dtype=np.float64,
    )
    best_idx = int(np.argmin(errors))
    return errors, best_idx
