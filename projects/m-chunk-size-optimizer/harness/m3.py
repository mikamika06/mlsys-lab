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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_argmin": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import optimizer.chunking as ch
    good_opt = ch.optimize_chunk_size

    def bad_optimize(trace, sizes):
        worst_idx = 0
        max_c = -float('inf')
        for i, sz in enumerate(sizes):
            sv = ch.calculate_prefix_savings(trace, sz)
            c = ch.benchmark_serving_frontier(trace, sz, sv)
            if c > max_c:
                max_c = c
                worst_idx = i
        return worst_idx

    ch.optimize_chunk_size = bad_optimize

    try:
        if not _survives(path):
            out["catches_bad_argmin"] = 1.0
        else:
            out["_note"] = "tests passed even when argmin was swapped to argmax"
    finally:
        ch.optimize_chunk_size = good_opt

    return out
