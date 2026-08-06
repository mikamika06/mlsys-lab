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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_split": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {e}"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gpucache.multi as m
    orig = m.allocate_multi_model_memory

    def bad_alloc(total, reserved, fractions):
        net = total - reserved
        return [net + 1000 for _ in fractions]

    m.allocate_multi_model_memory = bad_alloc
    try:
        survived = _survives(path)
        out["catches_invalid_split"] = 0.0 if survived else 1.0
    finally:
        m.allocate_multi_model_memory = orig
    return out
