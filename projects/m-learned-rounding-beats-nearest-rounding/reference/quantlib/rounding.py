import numpy as np


def nearest_rounding(weights, scale, zero_point):
    q = np.round(weights / scale + zero_point)
    q = np.clip(q, -8, 7)
    dequant = (q - zero_point) * scale
    return dequant


def learned_rounding(weights, scale, zero_point, steps=50):
    v = weights / scale + zero_point
    q_base = np.floor(v)
    r = np.clip(v - q_base, 0.0, 1.0)
    best_r = r.copy()
    best_mse = np.mean((weights - ((q_base + best_r - zero_point) * scale)) ** 2)

    for _ in range(steps):
        grad = 2.0 * ((q_base + r - zero_point) * scale - weights) * scale
        r = r - 0.01 * grad
        r = np.clip(r, 0.0, 1.0)
        curr_mse = np.mean((weights - ((q_base + r - zero_point) * scale)) ** 2)
        if curr_mse < best_mse:
            best_mse = curr_mse
            best_r = r.copy()

    final_q = q_base + np.round(best_r)
    final_q = np.clip(final_q, -8, 7)
    return (final_q - zero_point) * scale
