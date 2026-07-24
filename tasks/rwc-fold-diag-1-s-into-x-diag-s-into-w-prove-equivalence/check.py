import numpy as np
from mlsys import scorers


def _oracle(W, X, s):
    W_fold = W * s[np.newaxis, :]
    X_fold = X / s[:, np.newaxis]
    Y_fold = W_fold @ X_fold
    ratio = np.max(np.abs(X)) / (np.max(np.abs(X_fold)) + 1e-12)
    return W_fold, X_fold, Y_fold, ratio


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]], dtype=np.float64),
            np.array([[2.0, -4.0], [1.0, 3.0], [-2.0, 5.0]], dtype=np.float64),
            np.array([2.0, 0.5, 4.0], dtype=np.float64),
        ),
        (
            np.array(
                [[1.5, 2.0], [-3.0, 4.0], [0.5, -2.5]], dtype=np.float64
            ),
            np.array([[7.0], [-1.0]], dtype=np.float64),
            np.array([0.25, 8.0], dtype=np.float64),
        ),
    ]

    max_err = 0.0
    ratio_ok = 1.0

    for W, X, s in cases:
        try:
            got = sol.fold_diag_scales(W, X, s)
            W_fold, X_fold, Y_fold, ratio = got
        except Exception:
            return {"max_abs_err": float("inf"), "range_reduction_ratio": 0.0}

        ref_W, ref_X, ref_Y, ref_ratio = _oracle(W, X, s)

        err = scorers.max_abs_err(ref_Y, np.asarray(Y_fold))
        max_err = max(max_err, err)

        if not np.isclose(float(ratio), float(ref_ratio), rtol=1e-10, atol=1e-12):
            ratio_ok = 0.0

        fold_err = max(
            scorers.max_abs_err(ref_W, np.asarray(W_fold)),
            scorers.max_abs_err(ref_X, np.asarray(X_fold)),
        )
        max_err = max(max_err, fold_err)

    return {
        "max_abs_err": float(max_err),
        "range_reduction_ratio": float(ratio_ok),
    }
