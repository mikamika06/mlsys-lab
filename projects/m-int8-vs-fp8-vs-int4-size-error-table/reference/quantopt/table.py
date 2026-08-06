import numpy as np


def quantize_dequantize_int8(x):
    max_val = float(np.max(np.abs(x)))
    if max_val == 0.0:
        return np.zeros_like(x)
    scale = max_val / 127.0
    q = np.clip(np.round(x / scale), -128, 127)
    return q * scale


def quantize_dequantize_fp8(x):
    max_val = float(np.max(np.abs(x)))
    if max_val == 0.0:
        return np.zeros_like(x)
    max_fp8 = 448.0
    scale = max_val / max_fp8
    scaled = np.clip(x / scale, -max_fp8, max_fp8)
    q = np.round(scaled / 16.0) * 16.0
    return q * scale


def quantize_dequantize_int4(x, group_size=64):
    shape = x.shape
    flat = x.reshape(-1)
    n = len(flat)
    out = np.zeros_like(flat)
    for i in range(0, n, group_size):
        chunk = flat[i : i + group_size]
        max_val = float(np.max(np.abs(chunk)))
        if max_val == 0.0:
            out[i : i + group_size] = 0.0
            continue
        scale = max_val / 7.0
        q = np.clip(np.round(chunk / scale), -8, 7)
        out[i : i + group_size] = q * scale
    return out.reshape(shape)


def build_precision_table(weights, activations):
    base_bytes = weights.size * 2 + activations.size * 2
    res = {}

    w_int8 = quantize_dequantize_int8(weights)
    a_int8 = quantize_dequantize_int8(activations)
    bytes_int8 = weights.size * 1 + activations.size * 1
    res["int8"] = {
        "size_bytes": bytes_int8,
        "size_ratio": float(bytes_int8 / base_bytes),
        "weight_error": float(np.mean((weights - w_int8) ** 2)),
        "act_error": float(np.mean((activations - a_int8) ** 2)),
        "total_error": float(
            np.mean((weights - w_int8) ** 2) + np.mean((activations - a_int8) ** 2)
        ),
    }

    w_fp8 = quantize_dequantize_fp8(weights)
    a_fp8 = quantize_dequantize_fp8(activations)
    bytes_fp8 = weights.size * 1 + activations.size * 1
    res["fp8"] = {
        "size_bytes": bytes_fp8,
        "size_ratio": float(bytes_fp8 / base_bytes),
        "weight_error": float(np.mean((weights - w_fp8) ** 2)),
        "act_error": float(np.mean((activations - a_fp8) ** 2)),
        "total_error": float(
            np.mean((weights - w_fp8) ** 2) + np.mean((activations - a_fp8) ** 2)
        ),
    }

    w_int4 = quantize_dequantize_int4(weights, group_size=64)
    bytes_int4 = (weights.size * 4) // 8 + activations.size * 2
    res["int4"] = {
        "size_bytes": bytes_int4,
        "size_ratio": float(bytes_int4 / base_bytes),
        "weight_error": float(np.mean((weights - w_int4) ** 2)),
        "act_error": 0.0,
        "total_error": float(np.mean((weights - w_int4) ** 2)),
    }

    return res
