import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_dequantize": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import nf4.quantize as q
        good = q.dequantize_blockwise

        def broken_dequantize(quantized, absmax, codebook, block_size=64):
            # Intentionally ignore absmax, scaling by 1.0 instead
            broken_absmax = np.ones_like(absmax)
            return good(quantized, broken_absmax, codebook, block_size)

        q.dequantize_blockwise = broken_dequantize
        try:
            out["catches_broken_dequantize"] = 0.0 if _survives(path) else 1.0
        finally:
            q.dequantize_blockwise = good
    except ImportError:
        pass

    return out
