import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_canary_test", path)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_layout_regression": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner test failed on valid code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import matrix.canary as canary_mod
    orig_validate = canary_mod.validate_canary_artifact

    def broken_layout_validate(candidate_trace, reference_trace, tolerance=1e-4):
        import numpy as np
        cand_data = np.array(candidate_trace.get("data", []), dtype=np.float32)
        ref_data = np.array(reference_trace.get("data", []), dtype=np.float32)
        if cand_data.shape != ref_data.shape:
            return False
        diff = np.abs(cand_data - ref_data)
        return float(np.max(diff)) <= tolerance

    canary_mod.validate_canary_artifact = broken_layout_validate
    import matrix
    matrix.canary.validate_canary_artifact = broken_layout_validate

    try:
        out["catches_layout_regression"] = 0.0 if _survives(path) else 1.0
    finally:
        canary_mod.validate_canary_artifact = orig_validate
        matrix.canary.validate_canary_artifact = orig_validate

    return out
