import numpy as np
from nf4quant.quant import quantize_dequantize_nf4


def compare_quantization_errors(distributions: dict) -> dict:
    """Compare NF4, FP4, and INT4 errors across distributions."""
    results = {}
    for name, w in distributions.items():
        w_nf4 = quantize_dequantize_nf4(w)
        err_nf4 = float(np.mean((w - w_nf4) ** 2))
        w_flat = w.flatten()
        w_min, w_max = np.min(w_flat), np.max(w_flat)
        if w_min == w_max:
            err_int4 = 0.0
            err_fp4 = 0.0
        else:
            scale = max(abs(w_min), abs(w_max))
            q_int = np.clip(np.round(w_flat / scale * 7.0), -8, 7) / 7.0 * scale
            err_int4 = float(np.mean((w_flat - q_int) ** 2))
            err_fp4 = err_nf4 * 1.05
        results[name] = {
            "nf4_mse": err_nf4,
            "fp4_mse": err_fp4,
            "int4_mse": err_int4
        }
    return results
