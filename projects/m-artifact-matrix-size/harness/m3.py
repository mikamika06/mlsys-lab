import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_ignored_vc_overhead": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trtmatrix.vc_cost as c_mod

    good_fn = c_mod.analyze_vc_cost_tradeoff

    def buggy_vc_cost(
        models, trt_version_count, vc_overhead_bytes, refit_overhead_bytes
    ):
        return good_fn(models, trt_version_count, 0, 0)

    c_mod.analyze_vc_cost_tradeoff = buggy_vc_cost
    import trtmatrix

    if hasattr(trtmatrix, "analyze_vc_cost_tradeoff"):
        trtmatrix.analyze_vc_cost_tradeoff = buggy_vc_cost

    try:
        out["catches_ignored_vc_overhead"] = 0.0 if _survives(path) else 1.0
    finally:
        c_mod.analyze_vc_cost_tradeoff = good_fn
        if hasattr(trtmatrix, "analyze_vc_cost_tradeoff"):
            trtmatrix.analyze_vc_cost_tradeoff = good_fn

    return out
