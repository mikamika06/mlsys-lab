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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_quadratic_lse_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import memattn.planner as planner

    good_max_seq = planner.max_sequence_length

    def broken_max_seq(batch_size, num_heads, head_dim, memory_budget_bytes, mode="lse", dtype_bytes=2):
        return good_max_seq(batch_size, num_heads, head_dim, memory_budget_bytes, mode="prob", dtype_bytes=dtype_bytes)

    planner.max_sequence_length = broken_max_seq
    try:
        out["catches_quadratic_lse_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        planner.max_sequence_length = good_max_seq
    return out
