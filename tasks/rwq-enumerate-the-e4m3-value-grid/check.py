import numpy as np


def _oracle():
    """Vectorised E4M3 grid oracle: enumerate all 256 codes, decode, dedupe."""
    codes = np.arange(256, dtype=np.int64)
    S = (codes >> 7) & 1
    E = (codes >> 3) & 0xF
    M = codes & 0x7

    sign = np.where(S == 0, 1.0, -1.0)
    normal = sign * np.ldexp(1.0 + M / 8.0, (E - 7).astype(np.int64))
    subnorm = sign * np.ldexp(M / 8.0, -6)
    val = np.where(E == 0, subnorm, normal)

    is_nan = (E == 15) & (M == 7)
    finite = val[~is_nan]
    values = np.unique(finite)  # sorted, dedupes +0.0/-0.0

    max_finite = float(np.max(values))
    min_subnormal = float(np.ldexp(1.0 / 8.0, -6))
    return values, int(values.shape[0]), max_finite, min_subnormal


def grade(sol, fx) -> dict:
    ref_values, ref_n, ref_max, ref_min = _oracle()

    try:
        got = sol.e4m3_value_grid()
        vals = np.asarray(got["values"], dtype=np.float64)
        n_finite = int(got["n_finite"])
        max_finite = float(got["max_finite"])
        min_subnormal = float(got["min_subnormal"])
    except Exception:
        return {"exact_match": 0.0}

    if vals.shape != ref_values.shape:
        return {"exact_match": 0.0}
    if not np.array_equal(vals, ref_values):
        return {"exact_match": 0.0}
    if n_finite != ref_n:
        return {"exact_match": 0.0}
    if max_finite != ref_max:
        return {"exact_match": 0.0}
    if min_subnormal != ref_min:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
