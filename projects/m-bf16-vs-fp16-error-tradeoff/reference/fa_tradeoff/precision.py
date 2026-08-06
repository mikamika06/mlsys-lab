import numpy as np


def compute_relative_error(data):
    data_f32 = data.astype(np.float32)
    data_fp16 = data_f32.astype(np.float16).astype(np.float32)
    diff = np.abs(data_f32 - data_fp16)
    denom = np.maximum(np.abs(data_f32), 1e-7)
    return float(np.mean(diff / denom))
