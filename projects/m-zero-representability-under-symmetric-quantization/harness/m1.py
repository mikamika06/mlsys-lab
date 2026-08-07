import ref
import numpy as np


def check(workdir):
    from quant.sym import compute_scale
    out = {"scales_matched": 0.0}
    ok = 0
    for t in ref.TENSORS:
        want = ref.compute_scale(t, -128, 127)
        got = compute_scale(t, -128, 127)
        if np.allclose(want, got):
            ok += 1
    out["scales_matched"] = float(ok)
    return out
