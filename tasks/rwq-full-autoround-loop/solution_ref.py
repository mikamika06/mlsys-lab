import math
import numpy as np


def autoround_block(W: np.ndarray, bits: int, steps: int, lr: float, seed: int):
    np.random.seed(seed)
    W_arr = np.asarray(W, dtype=np.float64)
    w_flat = [float(x) for x in W_arr.flat]
    n = len(w_flat)

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    max_val = 0.0
    for x in w_flat:
        ax = abs(x)
        if ax > max_val:
            max_val = ax
    scale = max_val / qmax

    r = [0.0] * n

    def quantize(r_vec):
        res = []
        for x, off in zip(w_flat, r_vec):
            val = math.floor(x / scale + off + 0.5)
            if val < qmin:
                val = float(qmin)
            elif val > qmax:
                val = float(qmax)
            res.append(val * scale)
        return res

    def calc_mse(q_vec):
        total = 0.0
        for q_val, x in zip(q_vec, w_flat):
            diff = q_val - x
            total += diff * diff
        return total / n

    best = quantize(r)
    best_mse = calc_mse(best)

    factor = 2.0 * scale / n

    for _ in range(steps):
        current = quantize(r)
        current_mse = calc_mse(current)
        if current_mse < best_mse:
            best = current
            best_mse = current_mse

        for i in range(n):
            grad_i = factor * (current[i] - w_flat[i])
            if grad_i > 0.0:
                s = 1.0
            elif grad_i < 0.0:
                s = -1.0
            else:
                s = 0.0
            r[i] -= lr * s

    current = quantize(r)
    current_mse = calc_mse(current)
    if current_mse < best_mse:
        best = current
        best_mse = current_mse

    return np.array(best, dtype=np.float64).reshape(W_arr.shape), float(best_mse)
