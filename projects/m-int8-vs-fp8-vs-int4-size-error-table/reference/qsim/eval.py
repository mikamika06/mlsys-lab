import numpy as np
from .quant import quantize_int8, quantize_int4, quantize_fp8

def build_table(weights: np.ndarray):
    mse_int8 = np.mean((weights - quantize_int8(weights))**2)
    mse_int4 = np.mean((weights - quantize_int4(weights))**2)
    mse_fp8 = np.mean((weights - quantize_fp8(weights))**2)

    return {
        "fp32": {"size_ratio": 1.0, "mse": 0.0},
        "int8": {"size_ratio": 0.25, "mse": float(mse_int8)},
        "fp8": {"size_ratio": 0.25, "mse": float(mse_fp8)},
        "int4": {"size_ratio": 0.125, "mse": float(mse_int4)}
    }
