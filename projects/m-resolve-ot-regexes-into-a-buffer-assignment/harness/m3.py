import importlib.util
import os

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_priority": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moeoff.resolver as r
    orig = r.resolve_ot_regexes

    def bad_resolve(tensors, overrides, default_buffer="GPU"):
        assignments = {}
        for name, size in tensors:
            assigned = default_buffer
            for pattern, buf in reversed(overrides):
                if __import__("re").search(pattern, name):
                    assigned = buf
            assignments[name] = assigned
        return assignments

    r.resolve_ot_regexes = bad_resolve
    import moeoff
    moeoff.resolver = r
    try:
        out["catches_bad_priority"] = 0.0 if _survives(path) else 1.0
    finally:
        r.resolve_ot_regexes = orig
    return out
