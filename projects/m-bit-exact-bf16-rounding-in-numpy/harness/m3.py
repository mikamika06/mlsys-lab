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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_imprecise_rounding": 0.0}
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

    import bf16num.bf16 as bmod

    good_fn = bmod.fp32_to_bf16_bits

    def naive_fp32_to_bf16_bits(x):
        import numpy as np
        x_f32 = np.asarray(x, dtype=np.float32)
        u = x_f32.view(np.uint32)
        return (u >> 16).astype(np.uint16)

    bmod.fp32_to_bf16_bits = naive_fp32_to_bf16_bits
    if hasattr(sys.modules.get("bf16num"), "fp32_to_bf16_bits"):
        sys.modules["bf16num"].fp32_to_bf16_bits = naive_fp32_to_bf16_bits

    try:
        survived = _survives(path)
        out["catches_imprecise_rounding"] = 0.0 if survived else 1.0
    finally:
        bmod.fp32_to_bf16_bits = good_fn
        if hasattr(sys.modules.get("bf16num"), "fp32_to_bf16_bits"):
            sys.modules["bf16num"].fp32_to_bf16_bits = good_fn

    return out
