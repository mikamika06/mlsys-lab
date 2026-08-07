import numpy as np
from kvquant.quant import evaluate_quantization_error, compute_per_tensor_scale, quantize_fp8_e4m3, dequantize_fp8_e4m3


def identify_outlier_heads(x, threshold_std=2.0):
    head_maxes = np.max(np.abs(x), axis=(0, 2, 3))
    mean_val = np.mean(head_maxes)
    std_val = np.std(head_maxes)
    threshold = mean_val + threshold_std * std_val
    outliers = np.where(head_maxes > threshold)[0]
    return outliers.tolist()


def find_breaking_head(x, max_allowed_rel_err=0.15):
    scale = compute_per_tensor_scale(x)
    num_heads = x.shape[1]
    breaking_heads = []

    for h in range(num_heads):
        head_x = x[:, h, :, :]
        q = quantize_fp8_e4m3(head_x, scale)
        rec = dequantize_fp8_e4m3(q, scale)

        norm_diff = np.linalg.norm(head_x - rec)
        norm_orig = np.linalg.norm(head_x)
        rel_err = float(norm_diff / norm_orig) if norm_orig > 0 else 0.0

        if rel_err > max_allowed_rel_err:
            breaking_heads.append(h)

    return breaking_heads
