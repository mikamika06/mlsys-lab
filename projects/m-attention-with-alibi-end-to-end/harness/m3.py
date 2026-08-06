import importlib.util
import os
import numpy as np

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_softcap": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"test suite fails on good code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import alibi_attn.overflow as ov
    good_overflow = ov.measure_overflow_rate

    def broken_overflow(scores, threshold=65504.0, softcap=None):
        return good_overflow(scores, threshold=threshold, softcap=None)

    ov.measure_overflow_rate = broken_overflow
    import alibi_attn
    alibi_attn.overflow.measure_overflow_rate = broken_overflow

    try:
        out["catches_ignored_softcap"] = 0.0 if _survives(path) else 1.0
    finally:
        ov.measure_overflow_rate = good_overflow
        alibi_attn.overflow.measure_overflow_rate = good_overflow

    return out
