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
    import sys
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fake_quant": 0.0}

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import spec.quant as qmod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_quantize = qmod.quantize_weights

    def fake_quantize(w):
        return w, 0.0, 1.0

    qmod.quantize_weights = fake_quantize
    try:
        if not _survives(path):
            out["catches_fake_quant"] = 1.0
    finally:
        qmod.quantize_weights = good_quantize

    return out
