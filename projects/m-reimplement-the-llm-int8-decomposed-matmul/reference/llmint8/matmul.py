import numpy as np


def vector_scales(x):
    s = np.max(np.abs(x), axis=-1, keepdims=True)
    return np.maximum(s / 127.0, 1e-12)


def outlier_fraction_curve(x, thresholds):
    abs_x = np.abs(x)
    return [float(np.mean(abs_x > t)) for t in thresholds]


def decomposed_matmul(x, w_int8, w_scales, threshold):
    col_max = np.max(np.abs(x), axis=0)
    outlier_mask = col_max > threshold

    x_low = x[:, ~outlier_mask]
    w_low = w_int8[~outlier_mask, :]
    s_low = w_scales[~outlier_mask]

    x_high = x[:, outlier_mask] if np.any(outlier_mask) else np.zeros((x.shape[0], 0))
    w_high = w_int8[outlier_mask, :] if np.any(outlier_mask) else np.zeros((0, w_int8.shape[1]), dtype=w_int8.dtype)

    x_low_int8 = np.clip(np.round(x_low / vector_scales(x_low)), -127, 127).astype(np.int8) if x_low.size > 0 else np.zeros((x.shape[0], 0), dtype=np.int8)

    acc_low = np.dot(x_low_int8.astype(np.float64), w_low.astype(np.float64)) if x_low_int8.size > 0 else np.zeros((x.shape[0], w_int8.shape[1]))
    scale_matrix = vector_scales(x_low) * s_low.reshape(1, -1) if x_low.size > 0 else 0.0
    res_low = acc_low * scale_matrix if x_low.size > 0 else 0.0

    res_high = np.dot(x_high.astype(np.float64), w_high.astype(np.float64)) if x_high.size > 0 else 0.0

    return res_low + res_high
