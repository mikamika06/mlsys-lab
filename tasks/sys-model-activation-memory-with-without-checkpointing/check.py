import numpy as np

def _ref(layer_sizes, checkpoint_every):
    full = sum(np.zeros(sz, dtype=np.float64).nbytes for sz in layer_sizes)
    chkpt = sum(
        np.zeros(sz, dtype=np.float64).nbytes
        for i, sz in enumerate(layer_sizes)
        if i % checkpoint_every == 0
    )
    return full / chkpt

def grade(sol, fx) -> dict:
    cases = [
        ([10, 20, 30], 2),
        ([5, 5, 5, 5], 1),
        ([100, 200, 300, 400, 500], 3),
        ([7], 1),
        (list(range(1, 11)), 4)
    ]
    for sizes, ck in cases:
        try:
            got = sol.activation_memory_ratio(list(sizes), ck)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref(sizes, ck)
        rel_err = abs(got - ref) / (abs(ref) + 1e-12)
        if rel_err > 1e-9:
            return {"rel_err": rel_err}
    return {"rel_err": 0.0}
