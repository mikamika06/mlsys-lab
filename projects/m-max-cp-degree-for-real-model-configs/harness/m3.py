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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_degree": 0.0}
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

    import cpdegree.logeval as le
    orig = le.evaluate_throughput

    def broken_eval(config, cp_degree, base_tput):
        return base_tput * 2.0

    le.evaluate_throughput = broken_eval
    import cpdegree
    if hasattr(cpdegree, "evaluate_throughput"):
        cpdegree.evaluate_throughput = broken_eval

    try:
        out["catches_invalid_degree"] = 0.0 if _survives(path) else 1.0
    finally:
        le.evaluate_throughput = orig
        if hasattr(cpdegree, "evaluate_throughput"):
            cpdegree.evaluate_throughput = orig
    return out
