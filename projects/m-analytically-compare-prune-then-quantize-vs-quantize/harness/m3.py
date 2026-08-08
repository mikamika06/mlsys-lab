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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_order_fault": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import pqutils.pipeline as p
    orig_pipeline = p.run_joint_pipeline

    def broken_pipeline(weights, hessian, sparsity, q_bits, order):
        import numpy as np
        w = np.array(weights, dtype=np.float64)
        scale = (2.0 ** (q_bits - 1) - 1.0)
        return np.round(w * scale) / scale

    p.run_joint_pipeline = broken_pipeline
    import pqutils
    if hasattr(pqutils, "pipeline") and hasattr(pqutils.pipeline, "run_joint_pipeline"):
        pqutils.pipeline.run_joint_pipeline = broken_pipeline

    try:
        survives = _survives(path)
        out["catches_order_fault"] = 0.0 if survives else 1.0
        if survives:
            out["_note"] = "test suite failed to catch when prune and quantize order collapse to identical outputs"
    finally:
        p.run_joint_pipeline = orig_pipeline
        if hasattr(pqutils, "pipeline") and hasattr(pqutils.pipeline, "run_joint_pipeline"):
            pqutils.pipeline.run_joint_pipeline = orig_pipeline

    return out
