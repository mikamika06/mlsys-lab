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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaky_first_run": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ort_perf.profiler as prof
    good = prof.measure_breakdown

    def leaky(session_factory, inputs, time_fn, num_steady=5):
        t0 = time_fn()
        sess = session_factory()
        t1 = time_fn()
        for _ in range(num_steady + 1):
            sess.run(inputs)
        t2 = time_fn()
        avg = (t2 - t1) / (num_steady + 1)
        return {
            "creation": t1 - t0,
            "first_run": avg,
            "steady_step": avg
        }

    prof.measure_breakdown = leaky
    try:
        out["catches_leaky_first_run"] = 0.0 if _survives(path) else 1.0
    finally:
        prof.measure_breakdown = good

    return out
