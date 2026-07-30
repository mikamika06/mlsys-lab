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


def _swapped_unpack(packed, n):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    codes = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        byte = int(packed[i // 2])
        codes[i] = ((byte >> 4) & 0x0F) if i % 2 == 0 else (byte & 0x0F)
    return codes


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_swapped_nibbles": 0.0}
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

    import nibblepack.pack as p
    good = p.unpack_nibbles

    p.unpack_nibbles = _swapped_unpack
    import nibblepack
    nibblepack.unpack_nibbles = _swapped_unpack
    try:
        out["catches_swapped_nibbles"] = 0.0 if _survives(path) else 1.0
    finally:
        p.unpack_nibbles = good
        nibblepack.unpack_nibbles = good
    return out
