import importlib.util
import os

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_cache_key": 0.0, "catches_fake_timing": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import autotune.cache as acache
    import autotune.metrics as amet

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {e}"
        return out

    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_key = acache.make_cache_key
    acache.make_cache_key = lambda shape, stride: "static_key"
    try:
        out["catches_bad_cache_key"] = 0.0 if _survives(path) else 1.0
    finally:
        acache.make_cache_key = good_key

    good_measure = amet.measure_latency
    amet.measure_latency = lambda fn, args, warmup=10, reps=50: 0.0
    try:
        out["catches_fake_timing"] = 0.0 if _survives(path) else 1.0
    finally:
        amet.measure_latency = good_measure

    out["faults_caught"] = out["catches_bad_cache_key"] + out["catches_fake_timing"]
    return out
