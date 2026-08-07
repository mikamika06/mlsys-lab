import ref
import numpy as np

def check(workdir):
    from mlxops.reduction import measure_running_sum_error

    out = {
        "error_metrics_match": 0.0,
        "fp16_error_exceeds_fp32": 0.0,
    }

    try:
        got = measure_running_sum_error(ref.DATASET)
        want = ref.measure_running_sum_error(ref.DATASET)

        keys = ["max_err_fp32", "max_err_fp16", "final_err_fp32", "final_err_fp16", "drift_ratio"]
        all_match = True
        for k in keys:
            if not np.isclose(got.get(k, -1.0), want[k], rtol=1e-5, atol=1e-6):
                all_match = False
                out["_note"] = f"Key '{k}' mismatch: got {got.get(k)}, expected {want[k]}"
                break

        if all_match:
            out["error_metrics_match"] = 1.0

        if got.get("max_err_fp16", 0.0) > got.get("max_err_fp32", 0.0) * 5.0:
            out["fp16_error_exceeds_fp32"] = 1.0
        else:
            out["_note"] = "fp16 max error did not significantly exceed fp32 max error"

    except Exception as e:
        out["_note"] = f"Exception during reduction test: {type(e).__name__}: {e}"

    return out
