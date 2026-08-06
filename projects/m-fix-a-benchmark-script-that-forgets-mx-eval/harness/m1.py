import ref
import mlx.core as mx
import time


def check(workdir):
    out = {"evals_fixed": 0.0, "timing_nonzero": 0.0}
    try:
        from mlx_bench.bench import run_benchmark
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    mx.random.seed(42)
    x = mx.random.normal((256, 256))
    weights = [mx.random.normal((256, 256)) for _ in range(5)]

    start = time.perf_counter()
    res = run_benchmark(x, weights)
    mx.eval(res)
    duration = time.perf_counter() - start

    if res is not None:
        out["evals_fixed"] = 1.0
    if duration > 0.0:
        out["timing_nonzero"] = 1.0
    return out
