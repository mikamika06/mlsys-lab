import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    # Test on a few sequence lengths
    seq_lengths = [5, 10, 20]
    max_error = 0.0
    for n in seq_lengths:
        try:
            got = sol.create_causal_mask(n)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = np.tril(np.ones((n, n), dtype=np.float64))
        got_arr = np.array(got, dtype=np.float64)
        err = max_abs_err(ref, got_arr)
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
