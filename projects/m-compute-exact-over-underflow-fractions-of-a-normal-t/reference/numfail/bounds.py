import math


def compute_overflow_underflow_fractions(mean: float, std: float, dtype: str) -> dict:
    """Compute exact overflow and underflow fractions for a normal distribution."""
    if dtype == "fp16":
        max_val = 65504.0
        min_pos_norm = 6.103515625e-5
    elif dtype == "bf16":
        max_val = 3.389531389231535e38
        min_pos_norm = 1.1754943508222875e-38
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if std <= 0.0:
        abs_m = abs(mean)
        ov = 1.0 if abs_m > max_val else 0.0
        un = 1.0 if 0.0 < abs_m < min_pos_norm else 0.0
        return {"overflow": ov, "underflow": un}

    def cdf(x):
        return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))

    overflow = (1.0 - cdf(max_val)) + cdf(-max_val)
    underflow = cdf(min_pos_norm) - cdf(-min_pos_norm)
    return {"overflow": float(overflow), "underflow": float(underflow)}
