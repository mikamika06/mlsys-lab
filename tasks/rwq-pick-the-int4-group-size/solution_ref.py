import numpy as np


def _group_quant_mse(W: np.ndarray, group_size: int, bits: int) -> float:
    qmax = (1 << (bits - 1)) - 1
    n = W.shape[0]
    num_blocks = n // group_size
    total_sq_err = 0.0
    for i in range(num_blocks):
        start = i * group_size
        block = W[start : start + group_size]
        amax_b = 0.0
        for x in block:
            ax = abs(x)
            if ax > amax_b:
                amax_b = ax
        scale_b = amax_b / qmax if amax_b > 0 else 1.0
        for x in block:
            code = round(x / scale_b)
            if code < -qmax:
                code = -qmax
            elif code > qmax:
                code = qmax
            recon = code * scale_b
            diff = recon - x
            total_sq_err += diff * diff
    return float(total_sq_err / n)


def pick_int4_group_size(W: np.ndarray, group_sizes=(32, 64, 128, 256),
                          bits: int = 4, lam: float = 0.02):
    """Pick the group_size minimizing mse(gs) + lam * (16.0 / gs).

    Returns (best_group_size, best_cost, costs), costs aligned with
    `group_sizes` order.
    """
    W = np.asarray(W, dtype=np.float64)
    costs_list = []
    for gs in group_sizes:
        mse = _group_quant_mse(W, gs, bits)
        overhead = 16.0 / gs
        costs_list.append(mse + lam * overhead)

    costs = np.array(costs_list, dtype=np.float64)
    best_idx = 0
    best_cost = costs_list[0]
    for i in range(1, len(costs_list)):
        if costs_list[i] < best_cost:
            best_cost = costs_list[i]
            best_idx = i

    return int(group_sizes[best_idx]), float(best_cost), costs
