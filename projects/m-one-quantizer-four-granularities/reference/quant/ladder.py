import numpy as np
from quant.core import get_view, restore_view, calc_qparams, apply_quant, apply_dequant

def evaluate_ladder(w, group_size=32):
    results = []
    for gran in ["tensor", "axis_0", "axis_1", "group"]:
        w_view = get_view(w, gran, group_size)
        scale, zp = calc_qparams(w_view, symmetric=False)
        q_view = apply_quant(w_view, scale, zp, symmetric=False)
        w_approx_view = apply_dequant(q_view, scale, zp)
        w_approx = restore_view(w_approx_view, w.shape, gran)

        meta_bytes = scale.size * 2 + zp.size * 1
        max_abs_err = float(np.max(np.abs(w - w_approx)))

        results.append({
            "granularity": gran,
            "meta_bytes": meta_bytes,
            "max_abs_err": max_abs_err
        })
    return results
