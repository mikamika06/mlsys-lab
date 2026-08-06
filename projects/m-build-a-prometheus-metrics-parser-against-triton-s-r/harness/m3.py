import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_label_agg": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import triton_metrics.aggregator as agg
    orig_fn = agg.compute_model_request_summary

    def broken_agg(samples):
        res = {}
        for s in samples:
            if s.name == "nv_inference_request_success":
                model = "all_models_merged"
                if model not in res:
                    res[model] = {"success_count": 0.0, "avg_compute_time_ms": 0.0}
                res[model]["success_count"] += s.value
        return res

    agg.compute_model_request_summary = broken_agg
    try:
        survived = _survives(path)
        out["catches_broken_label_agg"] = 0.0 if survived else 1.0
    finally:
        agg.compute_model_request_summary = orig_fn

    return out
