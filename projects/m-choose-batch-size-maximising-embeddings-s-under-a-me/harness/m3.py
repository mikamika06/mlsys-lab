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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_truncation_errors": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import embedopt.truncation as t
    good_fn = t.truncate_sequence

    def buggy_truncate(tokens, num_ctx, policy="truncate_right"):
        tokens = list(tokens)
        length = len(tokens)
        if length > num_ctx and policy == "error":
            return {
                "tokens": tokens[:num_ctx],
                "truncated": True,
                "original_length": length,
                "final_length": num_ctx,
                "policy_applied": policy
            }
        return good_fn(tokens, num_ctx, policy)

    t.truncate_sequence = buggy_truncate
    import embedopt
    embedopt.truncate_sequence = buggy_truncate

    try:
        out["catches_missing_truncation_errors"] = 0.0 if _survives(path) else 1.0
    finally:
        t.truncate_sequence = good_fn
        embedopt.truncate_sequence = good_fn

    return out
