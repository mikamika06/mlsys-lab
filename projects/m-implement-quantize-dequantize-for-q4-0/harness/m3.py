import importlib.util
import os
import sys
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_q51_packing": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests fail on good codebase: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import qblocks.q5_1 as q51
    orig_dequant = q51.dequantize_q5_1

    def broken_dequant_q5_1(blocks: list[dict]) -> np.ndarray:
        out_arr = np.zeros(len(blocks) * 32, dtype=np.float32)
        for idx, b in enumerate(blocks):
            d = float(np.float16(b["d"]))
            m = float(np.float16(b["m"]))
            qs = b["qs"]
            for j in range(16):
                low = int(qs[j] & 0x0F)
                high = int((qs[j] >> 4) & 0x0F)
                out_arr[idx * 32 + j] = low * d + m
                out_arr[idx * 32 + j + 16] = high * d + m
        return out_arr

    q51.dequantize_q5_1 = broken_dequant_q5_1
    try:
        out["catches_broken_q51_packing"] = 0.0 if _survives(path) else 1.0
    finally:
        q51.dequantize_q5_1 = orig_dequant

    return out
