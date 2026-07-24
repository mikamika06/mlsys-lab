import numpy as np

def _reference(in_states, out_states):
    in_norms = np.linalg.norm(in_states, axis=1)
    out_norms = np.linalg.norm(out_states, axis=1)
    cos = np.sum(in_states * out_states, axis=1) / (in_norms * out_norms)
    return float(np.mean(cos))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for n, d in [(10, 64), (20, 128), (5, 32), (50, 256), (30, 100)]:
        a = rng.standard_normal((n, d))
        b = rng.standard_normal((n, d))
        ref = _reference(a, b)
        try:
            got = sol.mean_cosine_similarity(a, b)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, float):
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        max_err = max(max_err, err)
    return {"rel_err": max_err}
