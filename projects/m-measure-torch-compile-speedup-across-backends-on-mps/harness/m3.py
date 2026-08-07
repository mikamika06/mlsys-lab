import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_equivalence": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import compilebench.verify as v
    good = v.check_equivalence

    def broken(*args, **kwargs):
        return False

    v.check_equivalence = broken
    import compilebench
    compilebench.verify.check_equivalence = broken
    try:
        out["catches_broken_equivalence"] = 0.0 if _survives(path) else 1.0
    finally:
        v.check_equivalence = good
        compilebench.verify.check_equivalence = good
    return out
