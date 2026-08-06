import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath("."))


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_stack_mismatch": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import nsys_diag.nvtx as nvtx
    good_reconstruct = nvtx.reconstruct_nvtx_depths

    def broken_reconstruct(events):
        return [{"timestamp_ns": e["timestamp_ns"], "event_type": e["event_type"], "depth": 1} for e in events]

    nvtx.reconstruct_nvtx_depths = broken_reconstruct

    try:
        out["catches_stack_mismatch"] = 0.0 if _survives(path) else 1.0
    finally:
        nvtx.reconstruct_nvtx_depths = good_reconstruct
    return out
