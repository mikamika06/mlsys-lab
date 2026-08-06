import numpy as np


def naive_int8_matmul(x, w):
    scale = np.max(np.abs(w), axis=-1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    x_fp = x.astype(np.float32)
    w_fp = w_int8.astype(np.float32) * scale
    return np.matmul(x_fp, w_fp)


def decomposed_matmul(x, w, threshold):
    outlier_mask = np.abs(x) > threshold
    x_fp16 = np.where(outlier_mask, x, 0.0).astype(np.float32)
    x_int8_input = np.where(outlier_mask, 0.0, x)

    scale = np.max(np.abs(w), axis=-1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)

    res_int8 = np.matmul(x_int8_input.astype(np.float32), w_int8.astype(np.float32) * scale)
    res_fp16 = np.matmul(x_fp16, w.astype(np.float32))
    return res_int8 + res_fp16
