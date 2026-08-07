import ref
from calib.scales import compute_max_scale
import numpy as np


def check(workdir):
    out = {"scales_matched": 0.0}
    ok = 0
    for t in ref.TENSORS:
        abs_max = np.max(np.abs(t))
        want = float(abs_max / 127.0) if abs_max != 0 else 1.0
        got = compute_max_scale(t)
        if abs(got - want) < 1e-5:
            ok += 1
    out["scales_matched"] = float(ok)
    return out
