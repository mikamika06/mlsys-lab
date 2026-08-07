import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"scale_matches": 0.0, "transform_exact": 0.0}

    try:
        from smoothquant.scale import compute_migration_scales, apply_smoothquant
    except Exception as e:
        out["_note"] = f"Failed to import smoothquant.scale: {e}"
        return out

    act, weight = ref.generate_synthetic_model_data(123)
    X = act["layer_attn_q"]
    W = weight["layer_attn_q"]

    act_max = np.max(np.abs(X), axis=0)
    weight_max = np.max(np.abs(W), axis=0)
    alpha = 0.65

    want_scale = ref.ref_compute_migration_scales(act_max, weight_max, alpha)
    try:
        got_scale = compute_migration_scales(act_max, weight_max, alpha)
        if np.allclose(want_scale, got_scale, rtol=1e-4, atol=1e-5):
            out["scale_matches"] = 1.0
        else:
            out["_note"] = "Migration scale mismatch against reference oracle"
    except Exception as e:
        out["_note"] = f"compute_migration_scales raised exception: {e}"
        return out

    try:
        X_s, W_s = apply_smoothquant(X, W, got_scale)
        orig_prod = X @ W.T
        trans_prod = X_s @ W_s.T
        if np.allclose(orig_prod, trans_prod, rtol=1e-4, atol=1e-4):
            out["transform_exact"] = 1.0
        else:
            out["_note"] = "Scaled activation and weight product deviated from original FP32 output"
    except Exception as e:
        out["_note"] = f"apply_smoothquant raised exception: {e}"

    return out
