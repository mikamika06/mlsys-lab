import numpy as np
from mlsys.scorers import rel_err

INT32_MAX = 2147483647

def grade(sol, fx) -> dict:
    # Generate deterministic test cases
    rng = np.random.default_rng(0)
    shapes = [(10, 5), (100, 3), (50, 8)]
    all_rel_errs = []
    peak_ok = True
    for shape in shapes:
        X = rng.integers(0, 128, size=shape, dtype=np.uint8)

        # Reference computation using int64 to avoid overflow
        full_ref = np.sum(X, axis=0, dtype=np.int32)
        reduced_X = np.minimum(X, 63).astype(np.uint8)
        reduced_ref = np.sum(reduced_X, axis=0, dtype=np.int32)
        peak_int64 = np.max(np.cumsum(reduced_X.astype(np.int64), axis=0), axis=0)
        peak_ref = peak_int64.astype(np.int32)

        # Call solution
        try:
            full_sol, reduced_sol, peak_sol = sol.reduce_range_accumulator_safety(X)
        except Exception:
            return {"rel_err": 1e6, "peak_ok": False}

        # Validate shapes and dtypes
        if not (full_sol.shape == reduced_sol.shape == peak_sol.shape == full_ref.shape):
            return {"rel_err": 1e6, "peak_ok": False}
        if not (full_sol.dtype == reduced_sol.dtype == peak_sol.dtype == np.int32):
            return {"rel_err": 1e6, "peak_ok": False}

        # Compute relative error over all outputs
        ref_concat = np.concatenate([full_ref, reduced_ref, peak_ref]).astype(np.float64)
        sol_concat = np.concatenate([full_sol.astype(np.float64),
                                     reduced_sol.astype(np.float64),
                                     peak_sol.astype(np.float64)]).astype(np.float64)
        err = rel_err(ref_concat, sol_concat)
        all_rel_errs.append(err)

        # Peak safety check
        if not np.all(peak_int64 <= INT32_MAX):
            peak_ok = False

    overall_rel_err = float(max(all_rel_errs))
    return {"rel_err": overall_rel_err, "peak_ok": peak_ok}
