import sys
import numpy as np

sys.path.insert(0, ".")
from quant.scale import compute_max_scale, find_best_scale_mse, simulate_quant


def test_find_best_scale_improves_mse_on_outliers():
    np.random.seed(1337)
    w = np.random.normal(0, 1.0, 1500).astype(np.float32)
    w[0] = 12.5
    w[1] = -12.5

    best = find_best_scale_mse(w, num_candidates=50, qmin=-8, qmax=7)
    baseline = compute_max_scale(w, qmax=7)

    dq_best = simulate_quant(w, best, -8, 7)
    dq_base = simulate_quant(w, baseline, -8, 7)

    mse_best = np.mean((w - dq_best) ** 2)
    mse_base = np.mean((w - dq_base) ** 2)

    assert mse_best < mse_base
    assert best < baseline
