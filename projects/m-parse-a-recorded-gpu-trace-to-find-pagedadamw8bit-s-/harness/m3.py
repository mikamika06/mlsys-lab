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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_bounds": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        try:
            first = _run(path)
        except Exception as e:
            out["has_tests"] = 1.0
            out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
            return out

        if first is None:
            out["_note"] = "no test_* functions found"
            return out

        out["has_tests"] = 1.0
        out["passes_on_good"] = 1.0

        import parser.core as core
        good = core.calculate_spillover_bytes

        def broken_bounds(events, step_start, step_end):
            total = 0
            for e in events:
                if e["name"] in ("cudaMemcpyH2D", "cudaMemcpyD2H"):
                    total += e.get("args", {}).get("bytes", 0)
            return total

        core.calculate_spillover_bytes = broken_bounds
        try:
            out["catches_broken_bounds"] = 0.0 if _survives(path) else 1.0
        finally:
            core.calculate_spillover_bytes = good
    finally:
        sys.path.pop(0)

    return out
