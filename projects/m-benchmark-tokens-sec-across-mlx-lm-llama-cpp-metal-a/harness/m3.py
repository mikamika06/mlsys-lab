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
        "catches_flawed_benchmark": 0.0,
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
            f"tests fail on valid implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import benchedge.metrics as m

    good_fn = m.compute_benchmark_metrics

    def flawed_compute(
        backend,
        prompt_tokens,
        generated_tokens,
        t_start,
        t_first_token,
        t_end,
        rss_samples,
    ):
        ttft = max(0.0, t_first_token - t_start)
        flawed_decode_dur = max(0.0, t_end - t_start)
        tps = (
            generated_tokens / flawed_decode_dur
            if flawed_decode_dur > 0
            else 0.0
        )
        return m.BenchmarkResult(
            backend=backend,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_sec=round(ttft, 6),
            decode_duration_sec=round(flawed_decode_dur, 6),
            decode_tokens_per_sec=round(tps, 4),
            peak_rss_mb=round(max(rss_samples) if rss_samples else 0.0, 2),
        )

    m.compute_benchmark_metrics = flawed_compute
    import benchedge

    benchedge.metrics.compute_benchmark_metrics = flawed_compute
    try:
        out["catches_flawed_benchmark"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_benchmark_metrics = good_fn
        benchedge.metrics.compute_benchmark_metrics = good_fn

    return out
