import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learner_regression"] = mod
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_invariant": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import admission.sim

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct simulator: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_simulate = admission.sim.simulate

    def bad_simulate(trace, policy, max_len=0, max_wait=0):
        if policy == "time_limit":
            return good_simulate(trace, policy, max_len, max_wait + 100)
        return good_simulate(trace, policy, max_len, max_wait)

    admission.sim.simulate = bad_simulate

    try:
        survives = False
        try:
            survives = (_run(path) is True)
        except AssertionError:
            pass
        except Exception:
            pass

        out["catches_broken_invariant"] = 0.0 if survives else 1.0
        if survives:
            out["_note"] = "test did not fail when time_limit incorrectly allowed requests with wait > max_wait"
    finally:
        admission.sim.simulate = good_simulate

    return out
