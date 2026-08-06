import importlib.util
import os
import sys


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


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_metrics": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import slotplan.metrics as m
    good_compute = m.compute_cache_reuse_ratio

    def bad_compute(metrics_text):
        parsed = m.parse_metrics(metrics_text)
        cached = parsed.get("llamacpp:prompt_tokens_cached_total", 0.0)
        return cached

    m.compute_cache_reuse_ratio = bad_compute
    try:
        survived = False
        try:
            survived = (_run(path) is True)
        except Exception:
            survived = False
        out["catches_broken_metrics"] = 0.0 if survived else 1.0
    finally:
        m.compute_cache_reuse_ratio = good_compute

    return out
