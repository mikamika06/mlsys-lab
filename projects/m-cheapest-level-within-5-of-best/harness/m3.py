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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_tolerance_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ortopt.levels as lvl

    good_fn = lvl.select_cheapest_level

    def broken_select_cheapest_level(latencies, setup_costs, tolerance=0.05):
        best_lat = min(latencies)
        threshold = best_lat
        candidates = []
        for idx, (lat, cost) in enumerate(zip(latencies, setup_costs)):
            if lat <= threshold:
                candidates.append((cost, idx))
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][1]

    lvl.select_cheapest_level = broken_select_cheapest_level
    import ortopt

    ortopt.levels.select_cheapest_level = broken_select_cheapest_level

    try:
        survived = _survives(path)
        out["catches_tolerance_bug"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "Learner's tests passed even when tolerance calculation was broken"
    finally:
        lvl.select_cheapest_level = good_fn
        ortopt.levels.select_cheapest_level = good_fn

    return out
