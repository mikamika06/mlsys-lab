import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unhandled_failure": 0.0,
        "catches_ignored_fallback": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import pipeline.bls as bls

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_ft = bls.BLSOrchestrator.execute_with_fault_tolerance

    def broken_ft_no_error(self, initial_input, fallback_responses=None):
        return "BROKEN_SUPPRESSED_ERROR"

    bls.BLSOrchestrator.execute_with_fault_tolerance = broken_ft_no_error
    try:
        out["catches_unhandled_failure"] = 0.0 if _survives(path) else 1.0
    finally:
        bls.BLSOrchestrator.execute_with_fault_tolerance = orig_ft

    def broken_ft_no_fallback(self, initial_input, fallback_responses=None):
        for stage_name, stage in self.dag.stages.items():
            deps = self.dag.dependencies[stage_name]
            inp = initial_input if not deps else None
            stage.run(inp)
        return "NO_FALLBACK"

    bls.BLSOrchestrator.execute_with_fault_tolerance = broken_ft_no_fallback
    try:
        out["catches_ignored_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        bls.BLSOrchestrator.execute_with_fault_tolerance = orig_ft

    out["faults_caught"] = (
        out["catches_unhandled_failure"] + out["catches_ignored_fallback"]
    )
    return out
