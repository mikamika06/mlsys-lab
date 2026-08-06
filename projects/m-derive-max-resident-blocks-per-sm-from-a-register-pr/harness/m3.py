import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_rounding": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out[
            "_note"
        ] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import triton_calc.regs as r

    good_eff = r.effective_regs

    def broken_eff(regs, gran):
        return regs

    r.effective_regs = broken_eff
    import triton_calc

    if hasattr(triton_calc, "regs"):
        triton_calc.regs.effective_regs = broken_eff
    try:
        out["catches_bad_rounding"] = 0.0 if _survives(path) else 1.0
    finally:
        r.effective_regs = good_eff
        if hasattr(triton_calc, "regs"):
            triton_calc.regs.effective_regs = good_eff
    return out
