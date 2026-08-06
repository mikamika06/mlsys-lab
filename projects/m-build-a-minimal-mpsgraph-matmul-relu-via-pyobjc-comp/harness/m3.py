import importlib.util
import os
import sys


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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_flawed_benchmarks": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mpsgraph.benchmark as bm

    orig_bench = bm.benchmark_mps_vs_eager

    def flawed_bench(graph_fn, eager_fn, inputs, warmup=5, runs=20):
        return {
            "graph_latency_ms": 0.0,
            "eager_latency_ms": 1.0,
            "speedup": 999.0,
            "graph_p50_ms": 0.0,
            "graph_p95_ms": 0.0,
            "runs": 0,
            "warmup": 0,
        }

    bm.benchmark_mps_vs_eager = flawed_bench

    try:
        out["catches_flawed_benchmarks"] = 0.0 if _survives(path) else 1.0
    finally:
        bm.benchmark_mps_vs_eager = orig_bench
        if "learner_regression" in sys.modules:
            del sys.modules["learner_regression"]

    return out
