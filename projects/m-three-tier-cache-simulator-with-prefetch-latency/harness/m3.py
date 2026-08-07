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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_dirty_accounting_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    import cachesim.simulator as sim
    good_simulate = sim.simulate

    def bad_simulate(trace, l1_cap, l2_cap, policy="always", write_mode="wb"):
        res = good_simulate(trace, l1_cap, l2_cap, policy, write_mode)
        res_copy = dict(res)
        if res_copy["l2_evictions"] > 0 and write_mode == "wb":
            res_copy["write_penalty_ns"] = int(res_copy["write_penalty_ns"] * 0.5)
        return res_copy

    sim.simulate = bad_simulate

    try:
        survives = _survives(path)
        if not survives:
            out["catches_dirty_accounting_bug"] = 1.0
        else:
            out["_note"] = "tests passed even when L2 writeback penalty was under-reported"
    finally:
        sim.simulate = good_simulate

    return out
