import numpy as np


def get_test_fixtures():
    rng = np.random.default_rng(1337)
    fixtures = []
    for _ in range(5):
        w = rng.normal(loc=0.0, scale=1.5, size=128).astype(np.float32)
        im = rng.uniform(0.1, 10.0, size=128).astype(np.float32)
        fixtures.append((w, im))
    return fixtures


def optimal_scale(weights, imatrix, q_min, q_max):
    weights = np.asarray(weights, dtype=np.float32)
    imatrix = np.asarray(imatrix, dtype=np.float32)
    max_val = max(abs(q_min), abs(q_max))
    max_w = np.max(np.abs(weights))
    if max_w == 0:
        return 1.0
    initial_scale = max_w / max_val
    q = np.clip(np.round(weights / initial_scale), q_min, q_max)
    num = np.sum(imatrix * weights * q)
    den = np.sum(imatrix * q * q)
    if den == 0:
        return float(initial_scale)
    return float(num / den)


def weighted_round_to_nearest(weights, imatrix, q_min, q_max, steps=15):
    weights = np.asarray(weights, dtype=np.float32)
    imatrix = np.asarray(imatrix, dtype=np.float32)
    base_scale = optimal_scale(weights, imatrix, q_min, q_max)
    scales = np.linspace(base_scale * 0.8, base_scale * 1.2, steps)
    best_scale = base_scale
    min_error = float("inf")
    best_q = None

    for s in scales:
        if s <= 0:
            continue
        q = np.clip(np.round(weights / s), q_min, q_max)
        error = np.sum(imatrix * (weights - s * q) ** 2)
        if error < min_error:
            min_error = error
            best_scale = float(s)
            best_q = q

    return best_scale, best_q


def measure_gain(weights, imatrix, bit_widths):
    weights = np.asarray(weights, dtype=np.float32)
    imatrix = np.asarray(imatrix, dtype=np.float32)
    gains = {}
    unweighted_mse = {}
    weighted_mse = {}

    for bits in bit_widths:
        q_min = -(2 ** (bits - 1))
        q_max = (2 ** (bits - 1)) - 1
        max_w = np.max(np.abs(weights))
        max_q = max(abs(q_min), abs(q_max))
        s_unif = (max_w / max_q) if max_q > 0 and max_w > 0 else 1.0
        q_unif = np.clip(np.round(weights / s_unif), q_min, q_max)
        err_unif = np.sum(imatrix * (weights - s_unif * q_unif) ** 2)

        s_opt, q_opt = weighted_round_to_nearest(weights, imatrix, q_min, q_max)
        err_opt = np.sum(imatrix * (weights - s_opt * q_opt) ** 2)

        unweighted_mse[bits] = float(err_unif / np.sum(imatrix))
        weighted_mse[bits] = float(err_opt / np.sum(imatrix))
        gains[bits] = float(err_unif - err_opt)

    return {"unweighted_mse": unweighted_mse, "weighted_mse": weighted_mse, "gains": gains}
