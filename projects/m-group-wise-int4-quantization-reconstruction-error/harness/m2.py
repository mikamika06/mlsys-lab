import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"metrics_matched": 0.0, "mse_matches": 0.0}
    try:
        from quant.metrics import compute_reconstruction_mse, classify_saturation
    except Exception as e:
        out["_note"] = f"Failed to import metrics module: {type(e).__name__}: {e}"
        return out

    mse_ok = True
    metrics_ok = True

    for item in ref.TEST_DATASETS:
        t = item["tensor"]
        gs = item["group_size"]
        asym = item["asymmetric"]

        ref_q, ref_s, ref_zp = ref.quantize_group_int4(t, gs, asymmetric=asym)
        ref_rec = ref.dequantize_group_int4(ref_q, ref_s, ref_zp, gs)
        ref_mse = ref.compute_reconstruction_mse(t, ref_rec)
        ref_class = ref.classify_saturation(t, gs, asymmetric=asym)

        try:
            got_mse = compute_reconstruction_mse(t, ref_rec)
            got_class = classify_saturation(t, gs, asymmetric=asym)
        except Exception as e:
            out["_note"] = f"Function raised exception: {type(e).__name__}: {e}"
            return out

        if not np.isclose(ref_mse, got_mse, rtol=1e-5, atol=1e-5):
            mse_ok = False
            out["_note"] = f"MSE mismatch: expected {ref_mse}, got {got_mse}"
            break

        for k in ["mse", "saturated_count", "unsaturated_count", "total_count", "saturation_ratio"]:
            if k not in got_class:
                metrics_ok = False
                out["_note"] = f"Missing key {k} in classify_saturation output"
                break
            if isinstance(ref_class[k], float):
                if not np.isclose(ref_class[k], got_class[k], rtol=1e-4, atol=1e-4):
                    metrics_ok = False
                    out["_note"] = f"Mismatch in metric {k}: expected {ref_class[k]}, got {got_class[k]}"
                    break
            else:
                if ref_class[k] != got_class[k]:
                    metrics_ok = False
                    out["_note"] = f"Mismatch in metric {k}: expected {ref_class[k]}, got {got_class[k]}"
                    break

        if not metrics_ok:
            break

    if mse_ok:
        out["mse_matches"] = 1.0
    if metrics_ok:
        out["metrics_matched"] = 1.0

    return out
