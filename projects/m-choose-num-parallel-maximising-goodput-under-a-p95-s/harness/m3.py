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
        "catches_faulty_slo_filter": 0.0,
    }
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

    import runner.capacity as cap

    good_fn = cap.select_optimal_num_parallel

    def faulty_select(benchmark_results, p95_slo_ms):
        sorted_results = sorted(benchmark_results, key=lambda x: x["num_parallel"])
        best = sorted_results[-1]
        return {
            "num_parallel": best["num_parallel"],
            "max_goodput": len(best["latencies_ms"]) / best.get("duration_s", 1.0),
            "p95_latency_ms": 999.0,
        }

    cap.select_optimal_num_parallel = faulty_select
    import runner

    runner.capacity.select_optimal_num_parallel = faulty_select

    try:
        out["catches_faulty_slo_filter"] = 0.0 if _survives(path) else 1.0
    finally:
        cap.select_optimal_num_parallel = good_fn
        runner.capacity.select_optimal_num_parallel = good_fn

    return out
