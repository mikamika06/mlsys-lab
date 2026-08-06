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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_coarse_restriction": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moe.combinatorics as comb
    good_fn = comb.compare_combinations

    def broken_compare_combinations(coarse_cfg, fine_cfg):
        return {
            "coarse_combinations": 100,
            "fine_combinations": 10,
            "ratio": 0.1
        }

    comb.compare_combinations = broken_compare_combinations
    import moe
    moe.combinatorics.compare_combinations = broken_compare_combinations

    try:
        out["catches_coarse_restriction"] = 0.0 if _survives(path) else 1.0
    finally:
        comb.compare_combinations = good_fn
        moe.combinatorics.compare_combinations = good_fn

    return out
