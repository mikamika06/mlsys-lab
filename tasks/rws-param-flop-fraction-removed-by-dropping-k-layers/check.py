import numpy as np

def _ref(param_counts, k):
    total = param_counts.sum()
    if k <= 0:
        removed = 0
    else:
        removed = param_counts[:k].sum()
    return np.array([removed / total, (total - removed) / total], dtype=np.float64)

def grade(sol, fx) -> dict:
    cases = [
        (np.array([100, 200, 300]), 1),
        (np.array([10, 20, 30, 40]), 2),
        (np.array([5, 5, 5, 5, 5]), 0),
        (np.array([50, 60, 70, 80, 90]), 5),
    ]
    got = []
    ref = []
    for pc, k in cases:
        try:
            out = sol.removed_and_remaining(pc, k)
            if not isinstance(out, tuple) or len(out) != 2:
                return {"mse": float("inf")}
            got.append(np.array(out, dtype=np.float64))
            ref.append(_ref(pc, k))
        except Exception:
            return {"mse": float("inf")}
    got_arr = np.concatenate(got)
    ref_arr = np.concatenate(ref)
    mse_val = float(np.mean((got_arr - ref_arr) ** 2))
    return {"mse": mse_val}
