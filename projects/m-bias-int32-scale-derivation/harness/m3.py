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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_zp": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
        
    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        sys.path.pop(0)
        return out
        
    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out
        
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0
    
    import quantizer.conv as qc
    good_conv = qc.integer_conv2d
    
    def bad_conv(i_q, i_z, w_q, b_q):
        return good_conv(i_q, 0, w_q, b_q)
        
    qc.integer_conv2d = bad_conv
    
    try:
        out["catches_bad_zp"] = 0.0 if _survives(path) else 1.0
    finally:
        qc.integer_conv2d = good_conv
        sys.path.pop(0)
        
    return out
