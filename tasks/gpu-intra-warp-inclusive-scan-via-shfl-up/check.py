import numpy as np
from mlsys.sim import GPU


def grade(sol, fx) -> dict:
    N = 64                       # two warps, so a wrong scan crosses the boundary
    rng = np.random.default_rng(0)
    x = rng.random(N)

    # oracle: an inclusive prefix sum inside each 32-lane warp, computed here
    ref = np.concatenate([np.cumsum(x[i:i + 32]) for i in range(0, N, 32)])

    g = GPU(N, smem_size=64)
    g.gmem[:] = x
    try:
        m = g.launch(sol.inclusive_scan_warp, N // 32, 32, N)
    except Exception:
        return {"max_abs_err": float("inf"), "smem_waves": 1.0}

    return {"max_abs_err": float(np.max(np.abs(g.gmem - ref))),
            "smem_waves": float(m["smem_waves"])}
