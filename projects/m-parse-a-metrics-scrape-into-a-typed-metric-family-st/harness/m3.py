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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_counter_resets": 0.0,
    }
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

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

    import vllm_metrics.rates as r

    good_fn = r.compute_counter_rates

    def broken_compute_counter_rates(
        family1: dict, family2: dict, duration_seconds: float
    ) -> dict:
        results = {}
        if duration_seconds <= 0:
            return results
        s1_map = {
            frozenset(s["labels"].items()): s["value"]
            for s in family1.get("samples", [])
        }
        s2_map = {
            frozenset(s["labels"].items()): s["value"]
            for s in family2.get("samples", [])
        }
        for key, v2 in s2_map.items():
            if key in s1_map:
                v1 = s1_map[key]
                results[key] = (v2 - v1) / duration_seconds
        return results

    r.compute_counter_rates = broken_compute_counter_rates
    import vllm_metrics

    if hasattr(vllm_metrics, "compute_counter_rates"):
        vllm_metrics.compute_counter_rates = broken_compute_counter_rates

    try:
        out["catches_counter_resets"] = 0.0 if _survives(path) else 1.0
    finally:
        r.compute_counter_rates = good_fn
        if hasattr(vllm_metrics, "compute_counter_rates"):
            vllm_metrics.compute_counter_rates = good_fn

    return out
