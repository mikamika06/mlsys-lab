import numpy as np


def quantize_fp16(arr):
    return arr.astype(np.float16).astype(np.float32)


def quantize_int8_weights(weights):
    max_val = float(np.max(np.abs(weights)))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    dequant = q.astype(np.float32) * scale
    return q, scale, dequant


def quantize_int8_activations(x, rmin, rmax):
    if rmax == rmin:
        scale = 1.0
        zero_point = 0
    else:
        scale = float(rmax - rmin) / 255.0
        zero_point = int(np.round(-rmin / scale))
        zero_point = max(0, min(255, zero_point))
    q = np.clip(np.round(x / scale) + zero_point, 0, 255).astype(np.uint8)
    dequant = (q.astype(np.float32) - zero_point) * scale
    return q, scale, zero_point, dequant


def evaluate_mode_output(weights, bias, x, mode, calibration_range=None):
    if mode == "fp32":
        return x @ weights.T + bias
    elif mode == "fp16":
        w16 = quantize_fp16(weights)
        b16 = quantize_fp16(bias)
        x16 = quantize_fp16(x)
        return x16 @ w16.T + b16
    elif mode == "dynamic_int8":
        _, _, w_dequant = quantize_int8_weights(weights)
        return x @ w_dequant.T + bias
    elif mode == "full_int8":
        _, _, w_dequant = quantize_int8_weights(weights)
        if calibration_range is None:
            rmin, rmax = float(np.min(x)), float(np.max(x))
        else:
            rmin, rmax = calibration_range
        _, _, _, x_dequant = quantize_int8_activations(x, rmin, rmax)
        return x_dequant @ w_dequant.T + bias
    else:
        raise ValueError(f"Unknown mode: {mode}")
