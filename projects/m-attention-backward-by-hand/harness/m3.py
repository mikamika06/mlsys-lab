import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_non_deterministic_dropout": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import attnbwd.backward as bwd
    good_mask_fn = bwd.generate_dropout_mask

    def buggy_mask_fn(shape, p, seed):
        if p <= 0.0:
            return bwd.np.ones(shape, dtype=bwd.np.float64)
        return (bwd.np.random.rand(*shape) >= p).astype(bwd.np.float64)

    bwd.generate_dropout_mask = buggy_mask_fn
    import attnbwd
    attnbwd.backward.generate_dropout_mask = buggy_mask_fn
    try:
        out["catches_non_deterministic_dropout"] = 0.0 if _survives(path) else 1.0
    finally:
        bwd.generate_dropout_mask = good_mask_fn
        attnbwd.backward.generate_dropout_mask = good_mask_fn
    return out
