import numpy as np
from qsim.fp8 import quantize_to_fp8_vals

def get_scales(x, q_max, axis=1):
    m = np.max(np.abs(x), axis=axis, keepdims=True)
    m = np.maximum(m, 1e-9)
    return m / q_max

def quantize_int8(x, axis=1):
    scales = get_scales(x, 127.0, axis)
    q = np.clip(np.round(x / scales), -127.0, 127.0)
    return q * scales

def quantize_int4(x, axis=1):
    scales = get_scales(x, 7.0, axis)
    q = np.clip(np.round(x / scales), -7.0, 7.0)
    return q * scales

def quantize_fp8(x, axis=1):
    scales = get_scales(x, 448.0, axis)
    q = quantize_to_fp8_vals(x / scales)
    return q * scales

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

def calibrate_scales(acts, target_max=127.0):
    m = np.max(np.abs(acts), axis=0, keepdims=True)
    m = np.maximum(m, 1e-9)
    return m / target_max

def compare_domains(act_in, act_out, target_max=127.0):
    scale_in = calibrate_scales(act_in, target_max)
    scale_out = calibrate_scales(act_out, target_max)

    q_in = np.clip(np.round(act_in / scale_in), -target_max, target_max) * scale_in
    q_out = np.clip(np.round(act_in / scale_out), -target_max, target_max) * scale_out

    err_in = np.mean((act_in - q_in)**2)
    err_out = np.mean((act_in - q_out)**2)

    return float(err_in), float(err_out)

def detect_poison(acts, threshold_ratio=10.0):
    m = np.max(np.abs(acts), axis=0)
    p99 = np.percentile(np.abs(acts), 99, axis=0)
    p99 = np.maximum(p99, 1e-9)
    return bool(np.any(m > threshold_ratio * p99))
