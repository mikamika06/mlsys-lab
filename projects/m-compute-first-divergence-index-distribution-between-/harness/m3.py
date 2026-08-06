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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_blind_divergence": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import divergence.analyze as da
    except ImportError:
        return out

    good = da.compute_divergences

    def blind_divergences(set_a, set_b):
        return [-1] * len(set_a)

    da.compute_divergences = blind_divergences
    try:
        survives = False
        try:
            survives = (_run(path) is True)
        except Exception:
            pass
        out["catches_blind_divergence"] = 0.0 if survives else 1.0
    finally:
        da.compute_divergences = good

    return out
