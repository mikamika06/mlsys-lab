import numpy as np
from mlsys.scorers import max_abs_err

def _ref(weights, group_size, bits):
    w = np.asarray(weights, dtype=np.float64)
    n_groups = len(w) // group_size
    reshaped = w.reshape(n_groups, group_size)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    scale = (maxs - mins) / (2**bits - 1)
    bias = mins
    return scale.astype(np.float64), bias.astype(np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_error = 0.0
    try:
        for _ in range(5):
            # generate a random weight array whose length is a multiple of 64
            n_groups = rng.integers(1, 10)
            weights = rng.standard_normal(n_groups * 64).astype(np.float64)
            bits = 4
            group_size = 64

            ref_scale, ref_bias = _ref(weights, group_size, bits)

            # call student's function
            sol_scale, sol_bias = sol.compute_group_params(
                weights, group_size=group_size, bits=bits
            )

            # ensure shapes and dtypes are correct
            if (sol_scale.shape != ref_scale.shape or
                sol_bias.shape != ref_bias.shape):
                return {"max_abs_err": float("inf")}

            err = max(max_abs_err(ref_scale, sol_scale),
                      max_abs_err(ref_bias, sol_bias))
            if err > max_error:
                max_error = err

    except Exception:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": max_error}
