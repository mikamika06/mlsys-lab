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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_median": 0.0}

    if not os.path.isfile(path):
        return out

    try:
        from bakeoff import benchmark
    except Exception:
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

    good_run_stable = benchmark.Benchmark.run_stable

    def bad_run_stable(self, engine, prompt_tokens, gen_len, runs=3):
        results = []
        for _ in range(runs):
            results.append(self.run_perf(engine, prompt_tokens, gen_len))
        if not results:
            return {}
        return {k: max([r[k] for r in results]) for k in results[0].keys()}

    benchmark.Benchmark.run_stable = bad_run_stable
    try:
        out["catches_broken_median"] = 0.0 if _survives(path) else 1.0
    finally:
        benchmark.Benchmark.run_stable = good_run_stable

    return out
