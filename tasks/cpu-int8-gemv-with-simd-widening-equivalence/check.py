import numpy as np
from mlsys.sim import cache as cachesim
from mlsys import scorers


def _ref_gemv(A: np.ndarray, x: np.ndarray):
    """Reference int8 widening GEMV, row-major traversal."""
    m, n = A.shape
    y = np.zeros(m, dtype=np.int32)
    access = []
    base_A = 0
    base_x = m * n  # disjoint address space
    for i in range(m):
        acc = 0
        for j in range(n):
            # record addresses in the same order as loads
            access.append(base_A + i * n + j)
            access.append(base_x + j)
            acc += int(A[i, j]) * int(x[j])
        y[i] = acc
    return y, access


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    m, n = 8, 6
    A = rng.integers(-128, 127, size=(m, n), dtype=np.int8)
    x = rng.integers(-128, 127, size=(n,), dtype=np.int8)

    y_ref, acc_ref = _ref_gemv(A, x)

    try:
        y, acc = sol.int8_gemv(A.copy(), x.copy())
    except Exception as e:
        return {"max_abs_err": 1e9, "miss_rate_ok": 0.0}

    y = np.asarray(y, dtype=np.int32)
    err = scorers.max_abs_err(y_ref, y)

    # simulate cache: 4-way, 64 sets, 64B lines
    params = dict(line_bytes=64, sets=64, ways=4)
    ref_miss = cachesim.simulate(acc_ref, **params)["miss_rate"]
    cand_miss = cachesim.simulate(acc, **params)["miss_rate"]

    miss_rate_ok = 1.0 if cand_miss <= ref_miss * 1.05 else 0.0
    return {"max_abs_err": err, "miss_rate_ok": miss_rate_ok}
