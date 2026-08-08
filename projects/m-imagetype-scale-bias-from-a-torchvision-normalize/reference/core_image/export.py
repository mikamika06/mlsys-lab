def simulate_conversion(weights, precision="fp32"):
    out = {}
    scale_factor = 0.5 if precision == "fp16" else 1.0
    out["weights"] = [w * scale_factor for w in weights]
    out["precision"] = precision
    out["size_bytes"] = len(weights) * (2 if precision == "fp16" else 4) + 1024
    return out
