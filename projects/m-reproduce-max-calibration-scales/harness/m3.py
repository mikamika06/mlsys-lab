import importlib.util
import os
import sys
import numpy as np


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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_corrupted_block_scales": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"The learner's tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import calib.nvfp4 as nvfp4_mod

    orig_dequant = nvfp4_mod.dequantize_nvfp4_block

    def broken_dequantize_nvfp4_block(codes, scales, block_size=16):
        global_scale = np.max(scales) if scales.size > 0 else 1.0
        corrupted_scales = np.full_like(scales, global_scale)
        return orig_dequant(codes, corrupted_scales, block_size=block_size)

    nvfp4_mod.dequantize_nvfp4_block = broken_dequantize_nvfp4_block

    try:
        out["catches_corrupted_block_scales"] = 0.0 if _survives(path) else 1.0
    finally:
        nvfp4_mod.dequantize_nvfp4_block = orig_dequant

    return out
