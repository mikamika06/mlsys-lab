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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_placements": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner test failed on valid code: {e}"
        return out

    if first is None:
        out["_note"] = "No test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ktrans.placement as pl
    good_fn = pl.reconstruct_placement

    def broken_placement(num_layers, num_experts, expert_bytes, vram_budget, frequency_log):
        res = {}
        for l in range(num_layers):
            res[l] = {
                "gpu": list(range(num_experts)),
                "cpu": []
            }
        return res

    pl.reconstruct_placement = broken_placement
    import ktrans
    ktrans.placement.reconstruct_placement = broken_placement

    try:
        catches = not _survives(path)
        out["catches_invalid_placements"] = 1.0 if catches else 0.0
    finally:
        pl.reconstruct_placement = good_fn
        ktrans.placement.reconstruct_placement = good_fn

    return out
