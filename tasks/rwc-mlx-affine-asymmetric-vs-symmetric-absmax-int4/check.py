import numpy as np
from mlsys import scorers


def _affine_error(W):
    W = np.asarray(W, dtype=np.float64)
    lo = np.min(W)
    hi = np.max(W)
    if hi == lo:
        return float(np.mean((W - lo) ** 2))
    scale = (hi - lo) / 15.0
    zero = np.round(-lo / scale)
    q = np.clip(np.round(W / scale + zero), 0, 15)
    recon = scale * (q - zero)
    return float(np.mean((recon - W) ** 2))


def _symmetric_error(W):
    W = np.asarray(W, dtype=np.float64)
    scale = np.max(np.abs(W)) / 7.0
    if scale == 0:
        return float(np.mean(W ** 2))
    q = np.clip(np.round(W / scale), -8, 7)
    recon = scale * q
    return float(np.mean((recon - W) ** 2))


def _oracle(W):
    affine = _affine_error(W)
    symmetric = _symmetric_error(W)
    best = "affine" if affine <= symmetric else "symmetric"
    return np.array([affine, symmetric], dtype=np.float64), best


def grade(sol, fx) -> dict:
    cases = [
        np.array([-1.0, -0.5, 0.1, 0.2, 2.8], dtype=np.float64),
        np.array([-3.2, -2.9, -2.5, 0.1, 0.2, 0.4], dtype=np.float64),
        np.linspace(-1.0, 1.0, 17, dtype=np.float64),
        np.array([-0.03, 0.01, 0.02, 0.05, 0.07, 0.09], dtype=np.float64),
    ]
    worst = 0.0
    match = 1.0
    for W in cases:
        ref, scheme = _oracle(W)
        try:
            got = sol.compare_int4_quantizers(W)
            errs = np.asarray(got[:2], dtype=np.float64)
            chosen = got[2]
        except Exception:
            return {"rel_err": 1.0, "scheme_match": 0.0}
        worst = max(worst, scorers.rel_err(ref, errs))
        if chosen != scheme:
            match = 0.0
    return {"rel_err": worst, "scheme_match": match}
