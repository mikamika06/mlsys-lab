import importlib.util
import os
import sys
import ref


def _run(path, workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
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


def _survives(path, workdir):
    try:
        return _run(path, workdir) is True
    except Exception:
        return False


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_suboptimal_k": 0.0,
    }
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from specdec.optimize import find_optimal_k
    except Exception as e:
        out["_note"] = f"Import failed: {e}"
        return out

    for case in ref.M3_CASES:
        want_k, _ = ref.find_optimal_k(case["alpha"], case["c"], case["max_k"])
        got_k, _ = find_optimal_k(case["alpha"], case["c"], case["max_k"])
        if got_k != want_k:
            out["_note"] = f"find_optimal_k returned {got_k}, expected {want_k}"
            return out

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path, workdir)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import specdec.optimize as opt

    good = opt.find_optimal_k

    def broken_find_optimal_k(alpha, c, max_k):
        return max_k, {k: 1.0 for k in range(1, max_k + 1)}

    opt.find_optimal_k = broken_find_optimal_k
    import specdec

    specdec.optimize.find_optimal_k = broken_find_optimal_k

    try:
        out["catches_suboptimal_k"] = 0.0 if _survives(path, workdir) else 1.0
    finally:
        opt.find_optimal_k = good
        specdec.optimize.find_optimal_k = good

    return out
