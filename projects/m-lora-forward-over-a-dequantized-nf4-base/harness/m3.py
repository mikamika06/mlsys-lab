import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_mismatched_dtype_nan": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests fail on good reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import qlora.forward as fwd
    good_forward = fwd.lora_nf4_forward

    def broken_forward_nan(x, qweight, absmax, codebook, lora_a, lora_b, scaling, compute_dtype="float32", block_size=64):
        res = good_forward(x, qweight, absmax, codebook, lora_a, lora_b, scaling, compute_dtype, block_size)
        res[0, 0] = np.nan
        return res

    import numpy as np
    fwd.lora_nf4_forward = broken_forward_nan

    try:
        catches = not _survives(path)
        out["catches_mismatched_dtype_nan"] = 1.0 if catches else 0.0
    finally:
        fwd.lora_nf4_forward = good_forward

    return out
