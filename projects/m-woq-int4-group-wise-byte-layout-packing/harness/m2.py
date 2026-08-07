import ref
import numpy as np


def check(workdir):
    out = {"error_ratio_valid": 0.0}
    try:
        from woq.quant import compute_quant_error
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    weights = ref.generate_test_data()
    group_size = 32
    try:
        err_raw = compute_quant_error(weights, group_size, smoothed=False)
        err_smooth = compute_quant_error(weights, group_size, smoothed=True)
    except Exception as e:
        out["_note"] = f"execution failed: {e}"
        return out

    if isinstance(err_raw, float) and isinstance(err_smooth, float):
        if err_raw >= 0.0 and err_smooth >= 0.0:
            out["error_ratio_valid"] = 1.0
        else:
            out["_note"] = "negative quantization error values"
    else:
        out["_note"] = "compute_quant_error did not return floats"
    return out
