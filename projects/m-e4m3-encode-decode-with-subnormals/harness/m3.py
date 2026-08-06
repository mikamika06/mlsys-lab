import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_subnormals": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fp8util.quant as q
    good_decode = q.decode_e4m3

    def broken_decode(b):
        import numpy as np
        b = np.asarray(b, dtype=np.uint8)
        shape = b.shape
        b_flat = b.flatten()
        out_arr = np.zeros(b_flat.shape, dtype=np.float32)
        for i, byte in enumerate(b_flat):
            sign = -1.0 if (byte & 0x80) else 1.0
            e = (byte >> 3) & 0x0F
            m = byte & 0x07
            if e == 0:
                out_arr[i] = 0.0
            else:
                actual_e = e - 7
                out_arr[i] = sign * (1.0 + m / 8.0) * (2**actual_e)
        return out_arr.reshape(shape)

    q.decode_e4m3 = broken_decode
    import fp8util
    fp8util.decode_e4m3 = broken_decode

    try:
        out["catches_bad_subnormals"] = 0.0 if _survives(path) else 1.0
    finally:
        q.decode_e4m3 = good_decode
        fp8util.decode_e4m3 = good_decode

    return out
