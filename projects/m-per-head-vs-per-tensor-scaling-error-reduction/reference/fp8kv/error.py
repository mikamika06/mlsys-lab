import numpy as np
from fp8kv.quant import simulate_e4m3, get_per_tensor_scale, get_per_head_scale


def measure_rel_err(orig, approx):
    denom = np.mean(np.abs(orig))
    if denom == 0:
        return 0.0
    return float(np.mean(np.abs(orig - approx)) / denom)


def find_breaking_head(x, max_val=448.0):
    st = get_per_tensor_scale(x, max_val)
    sh = get_per_head_scale(x, max_val)

    qt = simulate_e4m3(x, st)
    qh = simulate_e4m3(x, sh)

    worst_diff = -1.0
    worst_head = -1

    for i in range(x.shape[1]):
        err_t = measure_rel_err(x[:, i, :], qt[:, i, :])
        err_h = measure_rel_err(x[:, i, :], qh[:, i, :])
        diff = err_t - err_h
        if diff > worst_diff:
            worst_diff = diff
            worst_head = i

    return worst_head
