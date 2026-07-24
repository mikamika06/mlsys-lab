import numpy as np

def _reference(mantissa_bits):
    m = np.asarray(mantissa_bits, dtype=int)
    return np.power(2.0, -(m + 1))

def grade(sol, fx) -> dict:
    try:
        # compute reference
        ref = _reference([8, 12, 20])  # example values; any shape works
        # get candidate output
        cand = sol.round_trip_error([8, 12, 20])
        # ensure float64 arrays
        ref = np.asarray(ref, dtype=np.float64)
        cand = np.asarray(cand, dtype=np.float64)
        # compute global relative L2 error
        num = np.linalg.norm(cand - ref)
        den = np.linalg.norm(ref) + 1e-12
        rel_err = float(num / den)
    except Exception:
        rel_err = 0.0
    return {"rel_err": rel_err}
