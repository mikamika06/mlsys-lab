import numpy as np

def grade(sol, fx) -> dict:
    # Deterministic test case
    W = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])
    X = np.array([[0.5, 1.], 
                  [1.5, 2.], 
                  [2.5, 3.]])
    s = 0.75
    try:
        result = sol.scale_invariant_product(W, X, s)
    except Exception:
        return {"max_abs_err": float("inf")}
    if not isinstance(result, tuple) or len(result) != 2:
        return {"max_abs_err": float("inf")}
    orig, scaled = result
    # Reference values using NumPy
    ref_orig = W @ X
    ref_scaled = (W * s) @ (X / s)
    err_orig = np.max(np.abs(orig - ref_orig))
    err_scaled = np.max(np.abs(scaled - ref_scaled))
    return {"max_abs_err": float(err_orig + err_scaled)}
