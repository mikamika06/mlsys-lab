import numpy as np

def grade(sol, fx) -> dict:
    # Deterministic test data
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000).astype(np.float32) * 10 + 5

    # Reference implementation
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        ref_scale = 1.0
        ref_zp = 128
    else:
        ref_scale = (max_val - min_val) / 255.0
        ref_zp = int(round(-min_val / ref_scale))
        ref_zp = np.clip(ref_zp, 0, 255)

    # Call the student's implementation
    try:
        scale, zp = sol.dynamic_activation_scale_zero_point(x)
    except Exception:
        return {"scale_rel_err": float("inf"), "zero_point_match": 0.0}

    # Compute metrics
    scale_rel_err = np.linalg.norm(scale - ref_scale) / (np.abs(ref_scale) + 1e-12)
    zero_point_match = 1.0 if zp == int(ref_zp) else 0.0

    return {"scale_rel_err": float(scale_rel_err), "zero_point_match": float(zero_point_match)}
