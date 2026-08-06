import importlib.util
import os
import unittest

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
            if suite.countTestCases() > 0:
                res = unittest.TextTestRunner().run(suite)
                return res.wasSuccessful()
        except Exception:
            pass
        return None
    for fn in fns:
        if callable(fn):
            fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_logic": 0.0}
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
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import profiler_bench.ranking as r
    good_ranking = r.rank_profiler_options

    def broken_ranking():
        return ["with_flops", "record_shapes", "profile_memory", "with_stack"]

    r.rank_profiler_options = broken_ranking
    import profiler_bench
    profiler_bench.rank_profiler_options = broken_ranking

    try:
        out["catches_broken_logic"] = 0.0 if _survives(path) else 1.0
    finally:
        r.rank_profiler_options = good_ranking
        profiler_bench.rank_profiler_options = good_ranking
    return out
