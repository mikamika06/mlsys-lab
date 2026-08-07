import numpy as np


def compare_quality(baseline_outputs, quantized_outputs):
    base = np.array(baseline_outputs, dtype=np.float32)
    quant = np.array(quantized_outputs, dtype=np.float32)
    mse = float(np.mean((base - quant) ** 2))
    max_diff = float(np.max(np.abs(base - quant)))
    return {"mse": mse, "max_diff": max_diff, "valid": mse < 0.05}
