import numpy as np
from mlsys import scorers


def _ref(W_int8, w_scales, x_uint8, x_scale, x_zp):
    W = W_int8.astype(np.float64)
    x_signed = x_uint8.astype(np.float64) - x_zp
    acc = W @ x_signed          # shape (m,)
    y = acc * w_scales.astype(np.float64) * x_scale
    return y.astype(np.float32)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    worst = 0.0
    cases = [
        (2, 2, 128, 0.1),
        (4, 8, 128, 0.05),
        (16, 32, 0, 0.02),
        (64, 128, 64, 0.01),
        (128, 64, 200, 0.005),
    ]
    for m, n, zp, x_scale in cases:
        W_int8 = rng.integers(-128, 128, size=(m, n)).astype(np.int8)
        w_scales = rng.random(m).astype(np.float32) * 0.1 + 0.001
        x_uint8 = rng.integers(0, 256, size=(n,)).astype(np.uint8)
        ref = _ref(W_int8, w_scales, x_uint8, float(x_scale), int(zp))
        try:
            got = np.asarray(
                sol.dequant_linear_output(W_int8.copy(), w_scales.copy(), x_uint8.copy(), float(x_scale), int(zp)),
                dtype=np.float32
            )
        except Exception:
            return {"rel_err": 1e9}
        err = scorers.rel_err(ref, got)
        if err > worst:
            worst = err
    return {"rel_err": worst}
