import numpy as np


def compute_cancellation_error(data):
    x32 = np.array(data, dtype=np.float32)
    e_x32 = np.mean(x32)
    e_x2_32 = np.mean(x32 ** 2)
    var_naive_32 = e_x2_32 - (e_x32 ** 2)
    x16 = x32.astype(np.float16).astype(np.float32)
    e_x16 = np.mean(x16)
    e_x2_16 = np.mean(x16 ** 2)
    var_naive_16 = e_x2_16 - (e_x16 ** 2)
    err = abs(var_naive_16 - var_naive_32) / (abs(var_naive_32) + 1e-7)
    return float(err)


def two_pass_variance(data):
    x = np.array(data, dtype=np.float32)
    mean = np.mean(x)
    var = np.mean((x - mean) ** 2)
    return float(var)
