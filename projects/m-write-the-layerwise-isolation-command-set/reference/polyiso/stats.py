import numpy as np


def compute_mae(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a_arr - b_arr)))


def compute_max_abs_diff(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a_arr - b_arr)))


def compute_rel_error(a, b, eps=1e-7):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a_arr - b_arr) / (np.abs(b_arr) + eps)))


def compute_snr(a, b, eps=1e-10):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    signal_pow = np.sum(b_arr ** 2)
    noise_pow = np.sum((a_arr - b_arr) ** 2)
    if noise_pow < eps:
        return float("inf")
    return float(10.0 * np.log10(signal_pow / (noise_pow + eps)))


def compute_polygraphy_stats(a, b):
    return {
        "mae": compute_mae(a, b),
        "max_abs_diff": compute_max_abs_diff(a, b),
        "rel_error": compute_rel_error(a, b),
        "snr_db": compute_snr(a, b)
    }
