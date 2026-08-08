CONFIGS = [
    {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    {"mean": [0.1307, 0.1307, 0.1307], "std": [0.3081, 0.3081, 0.3081]}
]

def compute_scale_bias(mean, std):
    scale = [1.0 / (255.0 * s) for s in std]
    bias = [-m / s for m, s in zip(mean, std)]
    return scale, bias

def simulate_conversion(weights, precision="fp32"):
    out = {}
    scale_factor = 0.5 if precision == "fp16" else 1.0
    out["weights"] = [w * scale_factor for w in weights]
    out["precision"] = precision
    out["size_bytes"] = len(weights) * (2 if precision == "fp16" else 4) + 1024
    return out

def compute_drift_and_ratio(fp32_outputs, fp16_outputs, fp32_size, fp16_size):
    max_diff = max(abs(a - b) for a, b in zip(fp32_outputs, fp16_outputs))
    ratio = fp16_size / float(fp32_size)
    return max_diff, ratio
