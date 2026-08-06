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

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_norms": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import effbpw.compute as ec
    except ImportError:
        out["_note"] = "could not import effbpw.compute"
        return out

    good = ec.compute_effective_bpw

    def broken_compute_effective_bpw(tensor_shapes, base_bpw):
        import math
        total_bits = 0.0
        total_params = 0
        for shape in tensor_shapes.values():
            params = math.prod(shape)
            total_params += params
            total_bits += params * base_bpw
        return total_bits / total_params if total_params > 0 else 0.0

    ec.compute_effective_bpw = broken_compute_effective_bpw

    try:
        if not _survives(path):
            out["catches_ignored_norms"] = 1.0
        else:
            out["_note"] = "The broken compute_effective_bpw implementation survived the tests."
    finally:
        ec.compute_effective_bpw = good

    return out
