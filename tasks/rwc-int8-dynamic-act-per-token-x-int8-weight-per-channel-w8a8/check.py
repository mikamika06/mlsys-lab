import numpy as np
from mlsys import scorers

def _reference(A, W):
    eps = 1e-12
    # activation scales per row
    scale_row = np.max(np.abs(A), axis=1)
    scale_row[scale_row < eps] = 1.0
    a_int = np.round(A / scale_row[:, None]).clip(-128, 127).astype(np.int8)

    # weight scales per column
    scale_col = np.max(np.abs(W), axis=0)
    scale_col[scale_col < eps] = 1.0
    w_int = np.round(W / scale_col[None, :]).clip(-128, 127).astype(np.int8)

    y_int32 = a_int.astype(np.int32) @ w_int.astype(np.int32)
    Y = y_int32.astype(np.float64) * (scale_row[:, None] * scale_col[None, :])
    return Y

def grade(sol, fx):
    rngs = [np.random.default_rng(i) for i in range(3)]
    cases = [
        (rngs[0].normal(size=(5, 7)), rngs[1].normal(size=(7, 4))),
        (rngs[2].uniform(-10, 10, size=(3, 6)), rngs[0].uniform(-5, 5, size=(6, 5))),
        (np.zeros((4, 8)), np.ones((8, 3)) * 0.1),
    ]
    errors = []
    for A, W in cases:
        try:
            got = sol.int8_dynamic_act_per_token_x_int8_weight_per_channel(A, W)
            ref = _reference(A, W)
        except Exception:
            return {"max_abs_err": float("inf")}
        err = scorers.max_abs_err(ref, got)
        errors.append(err)
    max_err = max(errors)
    return {"max_abs_err": max_err}
