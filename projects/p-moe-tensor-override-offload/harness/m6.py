import importlib.util
import os

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

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_broken_freq": 0.0, "catches_broken_constraints": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import moe_offload.offload as offload_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_measure = offload_mod.MoEOffloader.measure_frequencies
    def broken_measure(self, traces):
        return offload_mod.np.zeros(len(self.tensor_sizes))
    offload_mod.MoEOffloader.measure_frequencies = broken_measure
    try:
        out["catches_broken_freq"] = 0.0 if _survives(path) else 1.0
    finally:
        offload_mod.MoEOffloader.measure_frequencies = good_measure

    good_check = offload_mod.MoEOffloader.check_constraints
    def broken_check(self, offloaded, memory_budget, latency, max_latency):
        return True
    offload_mod.MoEOffloader.check_constraints = broken_check
    try:
        out["catches_broken_constraints"] = 0.0 if _survives(path) else 1.0
    finally:
        offload_mod.MoEOffloader.check_constraints = good_check

    out["faults_caught"] = out["catches_broken_freq"] + out["catches_broken_constraints"]
    return out
