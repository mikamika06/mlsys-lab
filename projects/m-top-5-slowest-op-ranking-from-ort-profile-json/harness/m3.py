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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_category": 0.0}
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

    import ortprof.category as cat_mod
    good_func = cat_mod.classify_categories

    def broken_func(events):
        return {"GEMM": 0.0, "Normalization": 0.0, "Attention": 0.0, "Elementwise": 0.0, "Reduction": 0.0, "Other": 0.0}

    cat_mod.classify_categories = broken_func
    import ortprof
    if hasattr(ortprof, "classify_categories"):
        ortprof.classify_categories = broken_func

    try:
        out["catches_broken_category"] = 0.0 if _survives(path) else 1.0
    finally:
        cat_mod.classify_categories = good_func
        if hasattr(ortprof, "classify_categories"):
            ortprof.classify_categories = good_func
    return out
