import importlib.util
import os


def _run(path):
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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_dropped_requests": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out[
            "_note"
        ] = f"tests fail on valid harness: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bench_serve.runner as runner

    good_run = runner.run_bench_serve

    def broken_run_bench_serve(requests, rate, num_workers=1):
        res = good_run(requests, rate, num_workers)
        if res["results"]:
            res["results"] = res["results"][:-2]
        return res

    runner.run_bench_serve = broken_run_bench_serve
    if "bench_serve.bundle" in os.sys.modules:
        import bench_serve.bundle as bundle

        bundle.run_bench_serve = broken_run_bench_serve

    try:
        out["catches_dropped_requests"] = 0.0 if _survives(path) else 1.0
    finally:
        runner.run_bench_serve = good_run
        if "bench_serve.bundle" in os.sys.modules:
            bundle.run_bench_serve = good_run

    return out
