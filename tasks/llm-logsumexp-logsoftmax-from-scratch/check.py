import numpy as np

def _ref_logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    """Stable reference implementation using NumPy only."""
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))
    else:
        m = np.max(x, axis=axis, keepdims=True)
        sum_exp = np.sum(np.exp(x - m), axis=axis, keepdims=False)
        return (m.squeeze(axis) + np.log(sum_exp))

def _ref_log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """Stable reference implementation using NumPy only."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return (x - m) - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))

def grade(sol, fx) -> dict:
    # Prepare a set of random test cases
    rng = np.random.default_rng(42)
    tests = [
        rng.standard_normal((8,)),                     # 1‑D
        rng.standard_normal((6,5)) * 10.0,              # 2‑D with larger values
        rng.standard_normal((4,3,2)),                  # 3‑D
        rng.standard_normal((5,))*-1000.0,             # very negative
        rng.standard_normal((5,))*1000.0,              # very positive
    ]

    max_err = 0.0

    for arr in tests:
        # logsumexp
        try:
            cand_ls = sol.logsumexp(arr)
            ref_ls = _ref_logsumexp(arr)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        if cand_ls.dtype != np.float64:
            max_err = float("inf")
            break
        err = np.max(np.abs(cand_ls - ref_ls))
        max_err = max(max_err, err)

        # log_softmax (default axis)
        try:
            cand_lsm = sol.log_softmax(arr)
            ref_lsm = _ref_log_softmax(arr)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        if cand_lsm.dtype != np.float64:
            max_err = float("inf")
            break
        err = np.max(np.abs(cand_lsm - ref_lsm))
        max_err = max(max_err, err)

    # test axis parameter for 2‑D array
    arr2d = rng.standard_normal((6,5)) * 10.0
    try:
        cand_ls_axis1 = sol.logsumexp(arr2d, axis=1)
        ref_ls_axis1 = _ref_logsumexp(arr2d, axis=1)
        err = np.max(np.abs(cand_ls_axis1 - ref_ls_axis1))
        max_err = max(max_err, err)

        cand_lsm_axis0 = sol.log_softmax(arr2d, axis=0)
        ref_lsm_axis0 = _ref_log_softmax(arr2d, axis=0)
        err = np.max(np.abs(cand_lsm_axis0 - ref_lsm_axis0))
        max_err = max(max_err, err)
    except Exception:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": max_err}
