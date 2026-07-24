import numpy as np

def _oracle(weights, mask, drop_frac):
    live_indices = np.nonzero(mask)[0]
    if len(live_indices) == 0:
        return mask.copy()
    k = int(np.floor(len(live_indices) * drop_frac))
    if k == 0:
        return mask.copy()
    live_abs = np.abs(weights[live_indices])
    sorted_idx = np.argsort(live_abs)
    indices_to_drop = live_indices[sorted_idx[:k]]
    new_mask = mask.copy()
    new_mask[indices_to_drop] = False
    return new_mask

def grade(sol, fx) -> dict:
    cases = [
        # deterministic small case
        (np.array([0.5, -1.2, 3.4, -0.7]), np.array([True, True, False, True]), 0.5),
        # all live, half drop
        (np.arange(10, dtype=float), np.ones(10, dtype=bool), 0.5),
        # random weights and mask
        (np.random.randn(20), np.random.rand(20) > 0.3, 0.3),
        # no live weights
        (np.array([1.0, -2.0]), np.zeros(2, dtype=bool), 0.4),
        # drop_frac zero
        (np.array([1.5, -0.5, 2.0]), np.array([True, True, True]), 0.0),
    ]
    ok = 1.0
    for weights, mask, drop_frac in cases:
        try:
            got = sol.drop_step_prune(weights, mask, drop_frac)
            if not isinstance(got, np.ndarray):
                ok = 0.0
                break
            expected = _oracle(weights, mask, drop_frac)
            if not np.array_equal(got, expected):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
