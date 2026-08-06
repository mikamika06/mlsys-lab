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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_ratio": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prof.model as m
    good = m.compute_comm_bound_ratio

    def broken_ratio(timings, model_params):
        step_time = float(timings["step_time"])
        msg_size = float(model_params["msg_size_bytes"])
        bandwidth = float(model_params["bandwidth_bytes_per_sec"])
        return (msg_size / bandwidth / step_time) * 100.0

    m.compute_comm_bound_ratio = broken_ratio
    try:
        out["catches_broken_ratio"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_comm_bound_ratio = good
    return out
