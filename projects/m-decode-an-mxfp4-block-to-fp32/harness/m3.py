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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_corrupt_scale": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mxfp4.decode as dec
    orig_decode = dec.decode_mxfp4_block

    def broken_decode_mxfp4_block(scale_e8m0: int, nibbles: np.ndarray) -> np.ndarray:
        scale_e8m0 = int(scale_e8m0) & 0xFF
        scale = 1.0
        nibbles_arr = np.asarray(nibbles, dtype=np.uint8) & 0x0F
        return (dec.E2M1_TABLE[nibbles_arr] * scale).astype(np.float32)

    dec.decode_mxfp4_block = broken_decode_mxfp4_block
    import mxfp4
    mxfp4.decode_mxfp4_block = broken_decode_mxfp4_block

    try:
        survived = _survives(path)
        out["catches_corrupt_scale"] = 0.0 if survived else 1.0
    finally:
        dec.decode_mxfp4_block = orig_decode
        mxfp4.decode_mxfp4_block = orig_decode

    return out
