import numpy as np


def compute_max_scale(w_chan, qmax=7.0):
    m = np.max(np.abs(w_chan))
    return max(m / qmax, 1e-9)


def simulate_quant(w_chan, scale, qmin=-8, qmax=7):
    q = np.round(w_chan / scale)
    q = np.clip(q, qmin, qmax)
    return q * scale


def find_best_scale_mse(w_chan, num_candidates=100, qmin=-8, qmax=7):
    max_scale = compute_max_scale(w_chan, qmax)
    best_scale = max_scale
    best_mse = float('inf')
    for alpha in np.linspace(0.1, 1.0, num_candidates):
        scale = max_scale * alpha
        dq = simulate_quant(w_chan, scale, qmin, qmax)
        mse = np.mean((w_chan - dq) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
    return best_scale
