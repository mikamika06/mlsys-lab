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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_blockwise": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quant.core as qcore

    orig_blockwise = qcore.quantize_int4_blockwise

    def broken_blockwise(w: np.ndarray, block_size: int = 32):
        q, scales = orig_blockwise(w, block_size)
        scales = scales * 2.0
        return q, scales

    qcore.quantize_int4_blockwise = broken_blockwise
    import quant

    quant.core.quantize_int4_blockwise = broken_blockwise

    try:
        out["catches_broken_blockwise"] = 0.0 if _survives(path) else 1.0
    finally:
        qcore.quantize_int4_blockwise = orig_blockwise
        quant.core.quantize_int4_blockwise = orig_blockwise

    return out
