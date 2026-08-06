import numpy as np


def find_q4_1_advantage_distribution(weight_bank):
    best_idx = -1
    min_diff = float("inf")
    for i, w in enumerate(weight_bank):
        arr = np.array(w, dtype=np.float32)
        w_min, w_max = np.min(arr), np.max(arr)
        scale_0 = (w_max - w_min) / 15.0 if w_max > w_min else 1.0
        q0 = np.round(np.clip(arr / scale_0, 0, 15))
        mse_0 = np.mean((arr - (q0 * scale_0)) ** 2)

        scale_1 = (w_max - w_min) / 15.0 if w_max > w_min else 1.0
        min_val = w_min
        q1 = np.round(np.clip((arr - min_val) / scale_1, 0, 15))
        mse_1 = np.mean((arr - (q1 * scale_1 + min_val)) ** 2)

        diff = mse_1 - mse_0
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    return int(best_idx)
