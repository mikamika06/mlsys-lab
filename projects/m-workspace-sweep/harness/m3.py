import importlib.util
import os
import sys

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

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_opt_s_violation": 0.0}
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

    import sweep.engine as engine
    good_plan = engine.plan_engine

    def bad_plan_engine(config, profile, workspace_limit):
        max_ws = 0
        total_lat = 0.0
        for layer in config["layers"]:
            best_lat = float('inf')
            chosen_ws = 0
            for t in layer["tactics"]:
                ws = t["base_ws"] + t["ws_factor"] * profile["opt_s"]
                if ws <= workspace_limit:
                    lat = t["base_lat"] + t["lat_factor"] * profile["opt_s"]
                    if lat < best_lat:
                        best_lat = lat
                        chosen_ws = ws
            if best_lat == float('inf'):
                return float('inf'), float('inf')
            total_lat += best_lat
            max_ws = max(max_ws, chosen_ws)
        return config["weights_memory"] + max_ws, total_lat

    engine.plan_engine = bad_plan_engine

    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            pass

        if not survives:
            out["catches_opt_s_violation"] = 1.0
        else:
            out["_note"] = "tests passed even when plan_engine uses opt_s for workspace limit"
    finally:
        engine.plan_engine = good_plan

    return out
