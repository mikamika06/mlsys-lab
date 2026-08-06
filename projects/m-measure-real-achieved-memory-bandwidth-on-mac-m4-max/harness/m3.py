import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_inverted_roofline": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import macroofline.roofline as r
    good_fit = r.fit_empirical_roofline

    def broken_fit(profile_data, peak_gbps=546.0):
        res = good_fit(profile_data, peak_gbps)
        for p in res["profiles"]:
            p["is_memory_bound"] = not p["is_memory_bound"]
        return res

    r.fit_empirical_roofline = broken_fit
    try:
        out["catches_inverted_roofline"] = 0.0 if _survives(path) else 1.0
    finally:
        r.fit_empirical_roofline = good_fit
    return out
