import numpy as np
from mlsys.scorers import mse

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    X_arr = rng.standard_normal((50, 4))
    Y_arr = rng.standard_normal((30, 4))

    X = X_arr.tolist()
    Y = Y_arr.tolist()

    try:
        got = sol.pairwise_l1_matrix(X, Y)
        ref = np.abs(X_arr[:, None, :] - Y_arr[None, :, :]).sum(axis=2).tolist()
    except Exception:
        return {"mse": float("inf")}

    got_arr = np.array(got)
    ref_arr = np.array(ref)

    if got_arr.shape != ref_arr.shape:
        return {"mse": float("inf")}
    err = mse(ref_arr, got_arr)
    return {"mse": err}
