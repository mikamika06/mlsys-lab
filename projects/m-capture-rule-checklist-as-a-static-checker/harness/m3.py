import importlib.util
import os
import torch
import torch.fx

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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_uncaught_sync": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference implementation: {e}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import graph_checker.checker as orig_checker
    good_fn = orig_checker.check_graph_violations

    def broken_check_graph_violations(gm):
        res = good_fn(gm)
        return [r for r in res if r.get("rule") != "H2D_TRANSFER"]

    orig_checker.check_graph_violations = broken_check_graph_violations
    
    try:
        test_failed = False
        try:
            _run(path)
        except Exception:
            test_failed = True
        out["catches_uncaught_sync"] = 1.0 if test_failed else 0.0
    finally:
        orig_checker.check_graph_violations = good_fn

    return out
