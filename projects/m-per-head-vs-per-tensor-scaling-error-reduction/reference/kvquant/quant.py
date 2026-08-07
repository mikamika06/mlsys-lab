import numpy as np

E4M3_MAX = 448.0


def quantize_fp8_e4m3(x, scale):
    scaled = x * scale
    clipped = np.clip(scaled, -E4M3_MAX, E4M3_MAX)
    return clipped


def dequantize_fp8_e4m3(q, scale):
    return q / scale


def compute_per_tensor_scale(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return 1.0
    return E4M3_MAX / max_val


def compute_per_head_scales(x):
    max_vals = np.max(np.abs(x), axis=(0, 2, 3), keepdims=True)
    scales = np.where(max_vals == 0, 1.0, E4M3_MAX / max_vals)
    return scales


def evaluate_quantization_error(x, per_head=False):
    if per_head:
        scales = compute_per_head_scales(x)
    else:
        scales = compute_per_tensor_scale(x)

    q = quantize_fp8_e4m3(x, scales)
    x_rec = dequantize_fp8_e4m3(q, scales)

    norm_diff = np.linalg.norm(x - x_rec)
    norm_orig = np.linalg.norm(x)
    if norm_orig == 0:
        return 0.0
    return float(norm_diff / norm_orig)
