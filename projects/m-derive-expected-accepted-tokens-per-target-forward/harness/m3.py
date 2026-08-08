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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_faulty_throughput": 0.0,
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

    import specdec.tuning as t

    orig_eval = t.evaluate_draft_throughput

    def faulty_eval(acceptance_rates, draft_time, target_time):
        from specdec.metrics import expected_total_tokens

        results = {}
        for gamma in range(1, len(acceptance_rates) + 1):
            exp_total = expected_total_tokens(acceptance_rates[:gamma])
            total_latency = target_time
            results[gamma] = exp_total / total_latency
        return results

    t.evaluate_draft_throughput = faulty_eval
    try:
        out["catches_faulty_throughput"] = 0.0 if _survives(path) else 1.0
    finally:
        t.evaluate_draft_throughput = orig_eval

    return out
