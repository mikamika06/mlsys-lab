import ref
import numpy as np


def check(workdir):
    from quantlib.gidx import apply_gidx, invert_gidx

    rng = np.random.default_rng(42)
    w = rng.random((2, 8))
    g_idx = np.array([2, 0, 1, 3, 2, 0, 1, 3])

    try:
        p = apply_gidx(w, g_idx)
        back = invert_gidx(p, g_idx)
        if np.allclose(w, back):
            return {"gidx_matched": 1.0}
    except Exception as e:
        return {"gidx_matched": 0.0, "_note": str(e)}
    return {"gidx_matched": 0.0}
