import importlib.util
import os
import numpy as np

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_truncation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on correct solution: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no tests found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import dfkd.bounds as b
    good_fn = b.min_diversity_bound

    def buggy_bound(teacher_logits, rank, target_mse):
        V, d = teacher_logits.shape
        norms = np.sum(teacher_logits ** 2, axis=1)
        sorted_idx = np.argsort(-norms)
        for k in range(1, V + 1):
            unvis_err = np.sum(norms[sorted_idx[k:]])
            total_mse = unvis_err / (V * d)  # ignores truncation error
            if total_mse <= target_mse:
                return k
        return V

    b.min_diversity_bound = buggy_bound
    try:
        out["catches_ignored_truncation"] = 0.0 if _survives(path) else 1.0
    finally:
        b.min_diversity_bound = good_fn

    return out
