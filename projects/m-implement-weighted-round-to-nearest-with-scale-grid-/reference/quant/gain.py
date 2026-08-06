import numpy as np
from quant.round import weighted_round_to_nearest


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
