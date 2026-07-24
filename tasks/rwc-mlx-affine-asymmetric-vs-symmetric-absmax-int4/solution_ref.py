import numpy as np


def compare_int4_quantizers(W: np.ndarray) -> tuple[float, float, str]:
    W = np.asarray(W, dtype=np.float64)

    lo = np.min(W)
    hi = np.max(W)
    if hi == lo:
        affine_recon = np.full_like(W, lo)
    else:
        affine_scale = (hi - lo) / 15.0
        affine_zero = np.round(-lo / affine_scale)
        affine_q = np.clip(np.round(W / affine_scale + affine_zero), 0, 15)
        affine_recon = affine_scale * (affine_q - affine_zero)

    affine_err = float(np.mean((affine_recon - W) ** 2))

    sym_scale = np.max(np.abs(W)) / 7.0
    if sym_scale == 0:
        symmetric_recon = np.zeros_like(W)
    else:
        sym_q = np.clip(np.round(W / sym_scale), -8, 7)
        symmetric_recon = sym_scale * sym_q

    symmetric_err = float(np.mean((symmetric_recon - W) ** 2))
    best = "affine" if affine_err <= symmetric_err else "symmetric"

    return affine_err, symmetric_err, best
