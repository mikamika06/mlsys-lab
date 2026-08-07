import ref
import numpy as np

def check(workdir):
    from compressor_kv.oneshot import compute_kv_scales
    out = {"scales_computed": 0.0, "non_trivial_scales": 0.0}
    cfg = ref.CONFIGS[0]
    acts = ref.generate_calibration_data(cfg)
    try:
        res = compute_kv_scales(cfg, acts)
    except Exception as e:
        out["_note"] = f"raised exception: {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(res, (dict, list, float, np.floating)):
        out["scales_computed"] = 1.0

    scale_val = res["scale"] if isinstance(res, dict) else float(res)
    if scale_val != 1.0 and scale_val > 0.0:
        out["non_trivial_scales"] = 1.0
    else:
        out["_note"] = f"scale evaluated to trivial value: {scale_val}"
    return out
