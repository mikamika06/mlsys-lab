import ref
import numpy as np


def check(workdir):
    from quant.scales import compute_per_row_symmetric_scales
    weights = ref.generate_fixtures()
    max_err = 0.0
    for w in weights:
        max_vals = np.max(np.abs(w), axis=1, keepdims=True)
        want_scales = max_vals / 127.0
        want_scales = np.where(want_scales == 0, 1e-8, want_scales)
        got_scales = compute_per_row_symmetric_scales(w)
        err = float(np.max(np.abs(want_scales - got_scales)))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
