import numpy as np

from mlsys import scorers


def grade(sol, fx) -> dict:
    """
    Loads the fixed weight, activation, and Wanda pruning-mask fixtures,
    computes the pruned layer output Y = (W*M) @ X and the residual
    R = W@X - Y with a NumPy oracle, and compares the submission's Y and
    R (max abs error) to the oracle's.
    """
    W = np.asarray(fx["wanda_w"], dtype=np.float64)
    X = np.asarray(fx["wanda_x"], dtype=np.float64)
    M = np.asarray(fx["wanda_m"], dtype=np.float64)

    Y_exp = (W * M) @ X
    R_exp = W @ X - Y_exp

    try:
        Y_got, R_got = sol.apply_wanda_mask(W.copy(), M.copy(), X.copy())
        Y_got = np.asarray(Y_got, dtype=np.float64)
        R_got = np.asarray(R_got, dtype=np.float64)
    except Exception:
        return {"y_max_abs_err": float("inf"), "r_max_abs_err": float("inf")}

    y_err = scorers.max_abs_err(Y_exp, Y_got) if Y_got.shape == Y_exp.shape else float("inf")
    r_err = scorers.max_abs_err(R_exp, R_got) if R_got.shape == R_exp.shape else float("inf")

    return {"y_max_abs_err": y_err, "r_max_abs_err": r_err}
