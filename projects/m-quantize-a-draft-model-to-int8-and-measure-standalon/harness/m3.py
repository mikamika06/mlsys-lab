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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flawed_derivation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import draft.decide as d
    good_decide = d.is_net_end_to_end_win

    def naive_flawed_decide(latency_ratio, alpha_fp16, alpha_int8, target_latency, draft_fp16_latency, gamma=4):
        return latency_ratio < 1.0

    d.is_net_end_to_end_win = naive_flawed_decide
    import draft
    draft.decide.is_net_end_to_end_win = naive_flawed_decide

    try:
        out["catches_flawed_derivation"] = 0.0 if _survives(path) else 1.0
    finally:
        d.is_net_end_to_end_win = good_decide
        draft.decide.is_net_end_to_end_win = good_decide

    return out
