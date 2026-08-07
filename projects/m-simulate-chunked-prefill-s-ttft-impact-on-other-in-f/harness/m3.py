import os
import sys
import importlib.util

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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_monolithic": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good impl: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import llm_sched.scheduler as sch
    good_sim = sch.simulate_schedule

    def bad_sim(prompt, reqs, chunk, p_cost, d_cost):
        return good_sim(prompt, reqs, max(prompt, chunk) + 100, p_cost, d_cost)

    sch.simulate_schedule = bad_sim
    try:
        out["catches_monolithic"] = 0.0 if _survives(path) else 1.0
    finally:
        sch.simulate_schedule = good_sim

    return out
