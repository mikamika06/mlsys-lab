import importlib.util
import os
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_scales": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mlx_quant.unpack as unp
    good_fn = unp.unpack_and_dequantize_4bit

    def corrupt_unpack(packed_weights, scales, biases, group_size, original_shape):
        corrupted_scales = scales * 2.5
        return good_fn(packed_weights, corrupted_scales, biases, group_size, original_shape)

    unp.unpack_and_dequantize_4bit = corrupt_unpack
    import mlx_quant
    mlx_quant.unpack.unpack_and_dequantize_4bit = corrupt_unpack

    try:
        survived = _survives(path)
        out["catches_bad_scales"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "Regression tests failed to catch corrupted scale parameters"
    finally:
        unp.unpack_and_dequantize_4bit = good_fn
        mlx_quant.unpack.unpack_and_dequantize_4bit = good_fn

    return out
