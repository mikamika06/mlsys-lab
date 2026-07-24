import numpy as np

def _reference(structures):
    keys = sorted(structures.keys())
    mean_abs = np.empty(len(keys), dtype=np.float64)
    l2_norm = np.empty(len(keys), dtype=np.float64)
    taylor_imp = np.empty(len(keys), dtype=np.float64)
    for idx, k in enumerate(keys):
        act, grad, w = structures[k]
        act = np.asarray(act, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        mean_abs[idx] = np.mean(np.abs(act))
        l2_norm[idx] = np.linalg.norm(act)
        taylor_imp[idx] = np.mean(np.abs(grad * w))
    return mean_abs, l2_norm, taylor_imp

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    structures = {}
    for k in range(5):
        act = rng.standard_normal(size=10)
        grad = rng.standard_normal(size=10)
        w = rng.uniform(-1.0, 1.0)
        structures[k] = (act, grad, w)

    try:
        out = sol.importance_scores(structures)
    except Exception:
        return {"exact_match": 0.0}

    ref_mean_abs, ref_l2_norm, ref_taylor_imp = _reference(structures)

    if not isinstance(out, tuple) or len(out) != 3:
        return {"exact_match": 0.0}
    try:
        mean_abs, l2_norm, taylor_imp = out
    except Exception:
        return {"exact_match": 0.0}

    ok = (
        np.allclose(mean_abs, ref_mean_abs, rtol=0, atol=0) and
        np.allclose(l2_norm, ref_l2_norm, rtol=0, atol=0) and
        np.allclose(taylor_imp, ref_taylor_imp, rtol=0, atol=0)
    )
    return {"exact_match": 1.0 if ok else 0.0}
