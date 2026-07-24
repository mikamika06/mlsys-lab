import numpy as np

def _oracle(total_budget, num_layers):
    weights = np.arange(1, num_layers + 1)
    S = int(weights.sum())
    base = (total_budget * weights) // S
    remainder = total_budget - int(base.sum())
    # distribute remaining slots from bottom layer upwards
    for i in range(remainder):
        base[num_layers - 1 - i] += 1
    return base

def grade(sol, fx) -> dict:
    ok = 1.0
    try:
        for B, N in [(10, 4), (7, 4), (15, 6), (3, 5), (20, 8)]:
            got = sol.pyramidkv_allocation(B, N)
            ref = _oracle(B, N)
            if not isinstance(got, np.ndarray) or got.shape != ref.shape:
                ok = 0.0
                break
            if not np.array_equal(got, ref):
                ok = 0.0
                break
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
