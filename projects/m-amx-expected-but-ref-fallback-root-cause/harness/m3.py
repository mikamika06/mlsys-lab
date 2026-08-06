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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_masked_fallback": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference implementation: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import onednn_diag.fallback as fb
    orig_fb = fb.analyze_fallback_causes

    # Monkeypatch broken fallback analyzer that hides ref fallbacks
    def broken_fallback(logs):
        res = orig_fb(logs)
        return [r for r in res if "ref" not in r.get("implementation", "")]

    fb.analyze_fallback_causes = broken_fallback
    try:
        out["catches_masked_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        fb.analyze_fallback_causes = orig_fb

    return out
