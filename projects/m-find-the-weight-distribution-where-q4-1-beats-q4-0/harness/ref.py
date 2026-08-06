import numpy as np


def get_expected_distribution():
    np.random.seed(42)
    best_skew = None
    min_diff = float("inf")
    for skew in np.linspace(0.1, 5.0, 50):
        weights = np.random.exponential(scale=skew, size=256) - skew
        mse_q0 = _sim_q0(weights)
        mse_q1 = _sim_q1(weights)
        diff = mse_q1 - mse_q0
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_skew = float(skew)
    return {"skew": best_skew, "q4_1_better_at_scale": True}


def _sim_q0(w):
    d = (w.max() - w.min()) / 15.0
    if d == 0:
        d = 1e-5
    q = np.clip(np.round(w / d + 8), 0, 15)
    w_recon = (q - 8) * d
    return float(np.mean((w - w_recon) ** 2))


def _sim_q1(w):
    d = (w.max() - w.min()) / 15.0
    if d == 0:
        d = 1e-5
    m = w.min()
    q = np.clip(np.round((w - m) / d), 0, 15)
    w_recon = q * d + m
    return float(np.mean((w - w_recon) ** 2))
