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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_static_routing_violation": 0.0}
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

    import bls_router.router as r
    good_route = r.route_request

    def static_ensemble_broken(request, threshold=5.0):
        data = request["data"]
        res_a = data * 2.0
        res_b = data + 10.0
        return {"branch": "static_ensemble_all_branches", "result": res_a, "extra_branch_executed": res_b}

    r.route_request = static_ensemble_broken
    try:
        survived = False
        try:
            survived = _run(path) is True
        except Exception:
            survived = False
            
        out["catches_static_routing_violation"] = 0.0 if survived else 1.0
    finally:
        r.route_request = good_route
        
    return out
