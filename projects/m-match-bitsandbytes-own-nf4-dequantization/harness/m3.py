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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flipped_nibbles": 0.0}
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

    import nf4.dequant as dq
    good_unpack = dq.unpack_4bit
    good_dequantize = dq.dequantize_nf4

    def bad_unpack(packed_bytes):
        unpacked = np.empty(len(packed_bytes) * 2, dtype=np.uint8)
        unpacked[0::2] = packed_bytes >> 4
        unpacked[1::2] = packed_bytes & 0x0F
        return unpacked

    def bad_dequantize(packed_bytes, absmax, blocksize=64):
        unpacked = bad_unpack(packed_bytes)
        table = dq.get_nf4_table()
        expanded = np.repeat(absmax, blocksize)
        return table[unpacked] * expanded

    dq.unpack_4bit = bad_unpack
    dq.dequantize_nf4 = bad_dequantize

    try:
        out["catches_flipped_nibbles"] = 0.0 if _survives(path) else 1.0
    finally:
        dq.unpack_4bit = good_unpack
        dq.dequantize_nf4 = good_dequantize

    return out
