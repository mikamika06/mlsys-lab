import importlib.util
import os

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_sort": 0.0}

    if not os.path.isfile(path):
        return out

    import roofline.analysis as analysis

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_report = analysis.generate_report

    def bad_report(hw, kernels):
        res = good_report(hw, kernels)
        res.reverse()
        return res

    analysis.generate_report = bad_report
    try:
        out["catches_broken_sort"] = 0.0 if _survives(path) else 1.0
    finally:
        analysis.generate_report = good_report

    return out
