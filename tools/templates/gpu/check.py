import numpy as np
from mlsys.sim import GPU

def grade(sol, fx) -> dict:
    N, BLOCK, a = 256, 64, 3.0
    x = np.arange(N, dtype=np.float64)
    g = GPU(N)
    g.gmem[:] = x
    try:
        m = g.launch(sol.scale_kernel, N // BLOCK, BLOCK, N, a)
    except Exception:
        return {"max_abs_err": float("inf"), "transactions": 10**9}
    ref = a * x
    return {"max_abs_err": float(np.max(np.abs(g.gmem - ref))),
            "transactions": int(m["transactions"])}
