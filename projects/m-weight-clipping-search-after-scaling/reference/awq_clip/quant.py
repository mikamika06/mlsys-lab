import numpy as np


def quantize_and_reconstruct(w, max_val, n_bits=4):
    q_max = (1 << (n_bits - 1)) - 1
    scale = max_val / q_max
    scale = np.maximum(scale, 1e-7)
    w_q = np.clip(np.round(w / scale), -q_max, q_max)
    return w_q * scale


def search_clipping(w, n_bits=4, group_size=128, n_grid=100):
    w_reshaped = w.reshape(-1, group_size)
    num_groups = w_reshaped.shape[0]
    w_max = np.max(np.abs(w_reshaped), axis=1, keepdims=True)
    w_max = np.maximum(w_max, 1e-7)

    c_grid = np.linspace(0.01, 1.0, n_grid)
    errors = np.zeros((n_grid, num_groups))

    for i, c in enumerate(c_grid):
        cur_max = w_max * c
        w_rec = quantize_and_reconstruct(w_reshaped, cur_max, n_bits)
        err = np.sum((w_reshaped - w_rec)**2, axis=1)
        errors[i, :] = err

    best_idx = np.argmin(errors, axis=0)
    opt_max = w_max * c_grid[best_idx].reshape(-1, 1)

    return best_idx, opt_max
