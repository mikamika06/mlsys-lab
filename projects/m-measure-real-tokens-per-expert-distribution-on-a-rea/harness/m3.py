import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_starvation": 0.0}
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

    import moe_dist.metrics as m
    good_analyze = m.analyze_imbalance

    def broken_analyze(expert_counts, num_experts):
        res = good_analyze(expert_counts, num_experts)
        res["starved_experts"] = 0
        res["cv"] = 0.0
        return res

    m.analyze_imbalance = broken_analyze
    import moe_dist
    moe_dist.metrics.analyze_imbalance = broken_analyze

    try:
        out["catches_ignored_starvation"] = 0.0 if _survives(path) else 1.0
    finally:
        m.analyze_imbalance = good_analyze
        moe_dist.metrics.analyze_imbalance = good_analyze

    return out
