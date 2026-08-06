import numpy as np

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
