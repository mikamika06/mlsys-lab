import numpy as np


def analyze_float_formats(values):
    """Compute range, overflow, underflow, and ULP table across floating formats."""
    arr = np.asfarray(values)

    specs = {
        "FP32": {"exp": 8, "mant": 23, "max": 3.4028235e38, "min_pos": 1.1754944e-38},
        "TF32": {"exp": 8, "mant": 10, "max": 3.4028235e38, "min_pos": 1.1754944e-38},
        "FP16": {"exp": 5, "mant": 10, "max": 65504.0, "min_pos": 6.103515625e-5},
    }

    result = {}
    for fmt, s in specs.items():
        mant_bits = s["mant"]
        abs_vals = np.abs(arr)
        overflow = bool(np.any(abs_vals > s["max"]))
        underflow = bool(np.any((abs_vals > 0) & (abs_vals < s["min_pos"])))

        ulps = []
        for v in arr:
            if v == 0:
                bias = 127 if s["exp"] == 8 else 15
                ulp = 2.0 ** (-(mant_bits + bias - 1))
            else:
                exp_val = np.floor(np.log2(np.abs(v)))
                ulp = 2.0 ** (exp_val - mant_bits)
            ulps.append(float(ulp))

        result[fmt] = {
            "exponent_bits": s["exp"],
            "mantissa_bits": s["mant"],
            "max_value": float(s["max"]),
            "min_positive": float(s["min_pos"]),
            "overflow": overflow,
            "underflow": underflow,
            "max_ulp": float(np.max(ulps)) if len(ulps) > 0 else 0.0,
            "mean_ulp": float(np.mean(ulps)) if len(ulps) > 0 else 0.0,
        }

    return result
