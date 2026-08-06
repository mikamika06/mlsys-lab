import numpy as np


def quantize_dequantize_int4(x: np.ndarray, group_size: int):
    orig_shape = x.shape
    flat_x = x.astype(np.float64).reshape(-1)
    n = flat_x.shape[0]

    if n % group_size != 0:
        raise ValueError("Total elements must be divisible by group_size")

    num_groups = n // group_size
    grouped_x = flat_x.reshape(num_groups, group_size)

    max_vals = np.max(np.abs(grouped_x), axis=1, keepdims=True)
    scales = max_vals / 7.0
    scales[scales == 0.0] = 1.0

    q = np.clip(np.round(grouped_x / scales), -7, 7).astype(np.int8)
    dequant = (q.astype(np.float64) * scales).reshape(orig_shape)
    scales_out = scales.reshape(-1)

    return q.reshape(orig_shape), dequant, scales_out


def compute_reconstruction_mse(x: np.ndarray, group_size: int):
    _, dequant, _ = quantize_dequantize_int4(x, group_size)
    flat_orig = x.astype(np.float64).reshape(-1)
    flat_dequant = dequant.reshape(-1)

    diff = flat_orig - flat_dequant
    total_mse = float(np.mean(diff ** 2))

    num_groups = flat_orig.shape[0] // group_size
    group_diff = diff.reshape(num_groups, group_size)
    group_mse = np.mean(group_diff ** 2, axis=1).tolist()

    return {"total_mse": total_mse, "group_mse": group_mse}


def classify_elements(x: np.ndarray, group_size: int):
    orig_shape = x.shape
    flat_x = x.astype(np.float64).reshape(-1)
    num_groups = flat_x.shape[0] // group_size
    grouped_x = flat_x.reshape(num_groups, group_size)

    max_vals = np.max(np.abs(grouped_x), axis=1, keepdims=True)
    scales = max_vals / 7.0
    scales[scales == 0.0] = 1.0

    unclamped_q = grouped_x / scales
    is_clamped = np.abs(unclamped_q) > 7.0

    q = np.clip(np.round(unclamped_q), -7, 7)
    dequant = q * scales

    diff = grouped_x - dequant
    squared_err = diff ** 2

    clamped_mask = is_clamped
    in_range_mask = ~is_clamped

    total_clamped = int(np.sum(clamped_mask))
    total_in_range = int(np.sum(in_range_mask))
    total_elements = flat_x.shape[0]

    clamped_mse = float(np.sum(squared_err[clamped_mask])) / total_elements if total_elements > 0 else 0.0
    in_range_mse = float(np.sum(squared_err[in_range_mask])) / total_elements if total_elements > 0 else 0.0

    classification_mask = is_clamped.reshape(orig_shape)

    return {
        "clamped_mask": classification_mask,
        "clamped_count": total_clamped,
        "in_range_count": total_in_range,
        "clamped_ratio": float(total_clamped) / float(total_elements),
        "clamped_mse_contrib": clamped_mse,
        "in_range_mse_contrib": in_range_mse
    }
