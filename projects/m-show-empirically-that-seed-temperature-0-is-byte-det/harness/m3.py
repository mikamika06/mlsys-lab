import importlib.util
import os
import ref

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_throughput": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import det.profiler as p
    good_func = p.measure_decode_throughput

    def bad_throughput(temp):
        return good_func(temp) * (1.5 if temp > 0.5 else 0.5)

    p.measure_decode_throughput = bad_throughput
    try:
        out["catches_bad_throughput"] = 0.0 if _survives(path) else 1.0
    finally:
        p.measure_decode_throughput = good_func
    return out
