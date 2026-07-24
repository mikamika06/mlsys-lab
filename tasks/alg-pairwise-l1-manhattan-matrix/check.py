import numpy as np
from mlsys.scorers import mse

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    Y = rng.standard_normal((30, 4))

    try:
        got = sol.pairwise_l1_matrix(X, Y)
        ref = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
    except Exception:
        return {"mse": float("inf")}

    if got.shape != ref.shape:
        return {"mse": float("inf")}
    err = mse(ref, got)
    return {"mse": err}
