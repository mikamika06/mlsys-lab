import importlib.util
import os
import numpy as np

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_quant": 0.0, "catches_broken_scale": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import int4.quant as iq

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_quant = iq.quantize_weights
    iq.quantize_weights = lambda w, group_size=128: (np.zeros(w.size // 2, dtype=np.uint8), np.ones((w.size // group_size, 1), dtype=np.float32), w.shape)

    try:
        out["catches_bad_quant"] = 0.0 if _survives(path) else 1.0
    finally:
        iq.quantize_weights = good_quant

    def bad_quant(w, group_size=128):
        p, s, sh = good_quant(w, group_size)
        return p, s * 1000.0, sh

    iq.quantize_weights = bad_quant
    try:
        out["catches_broken_scale"] = 0.0 if _survives(path) else 1.0
    finally:
        iq.quantize_weights = good_quant

    out["faults_caught"] = out["catches_bad_quant"] + out["catches_broken_scale"]
    return out
