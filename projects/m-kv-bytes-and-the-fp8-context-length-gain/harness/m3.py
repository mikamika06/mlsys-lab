import importlib.util
import os


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_global_quant": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvfp8.quant as quant_mod

    good_quant = quant_mod.quantize_fp8_per_head

    def broken_global_quant(x):
        import numpy as np

        max_fp8 = 448.0
        scale = np.maximum(np.max(np.abs(x)) / max_fp8, 1e-12)
        q = np.clip(np.round(x / scale), -448.0, 448.0)
        return q, scale

    quant_mod.quantize_fp8_per_head = broken_global_quant
    try:
        out["catches_global_quant"] = 0.0 if _survives(path) else 1.0
    finally:
        quant_mod.quantize_fp8_per_head = good_quant
    return out
