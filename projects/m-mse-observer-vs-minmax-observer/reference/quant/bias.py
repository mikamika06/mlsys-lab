import numpy as np


def quantify_zero_point_bias(tensor, scale, zero_point, qmin, qmax):
    x = np.asarray(tensor, dtype=np.float64)
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax)

    deq_correct = (q - zero_point) * scale
    deq_ignored = q * scale

    bias_error = deq_ignored - deq_correct
    mean_bias = float(np.mean(bias_error))
    max_absolute_error = float(np.max(np.abs(deq_ignored - x)))
    correct_mse = float(np.mean((deq_correct - x) ** 2))
    ignored_mse = float(np.mean((deq_ignored - x) ** 2))

    return {
        "mean_bias": mean_bias,
        "max_absolute_error": max_absolute_error,
        "correct_mse": correct_mse,
        "ignored_mse": ignored_mse
    }
