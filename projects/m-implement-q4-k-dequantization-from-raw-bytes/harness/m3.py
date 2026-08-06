import ref
import os
import importlib.util

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
    from dequant import dequantize_q6_k

    out = {"q6_matches": 0.0, "has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_q6": 0.0}
    ok = 0
    for i, block in enumerate(ref.Q6_FIXTURES):
        want = ref.dequantize_q6_k(block)
        got = dequantize_q6_k(block)
        if len(want) == len(got) and max(abs(w - g) for w, g in zip(want, got)) < 1e-4:
            ok += 1
        elif "_note" not in out:
            diff = max(abs(w - g) for w, g in zip(want, got))
            out["_note"] = f"fixture {i}: max diff {diff}"

    out["q6_matches"] = float(ok)

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

    import dequant
    good = dequant.dequantize_q6_k

    def broken_q6_k(block):
        import struct
        ql = block[:128]
        qh = block[128:192]
        scales = struct.unpack('<16B', block[192:208])
        d = struct.unpack('<e', block[208:210])[0]
        y = [0.0] * 256
        for l in range(64):
            q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4)
            q2 = (ql[l] >> 4) | (((qh[l] >> 2) & 3) << 4)
            q3 = (ql[l + 64] & 0xF) | (((qh[l] >> 4) & 3) << 4)
            q4 = (ql[l + 64] >> 4) | (((qh[l] >> 6) & 3) << 4)
            y[l] = d * scales[l // 16] * (q1 - 32)
            y[l + 64] = d * scales[l // 16 + 4] * (q2 - 32)
            y[l + 128] = d * scales[l // 16 + 8] * (q3 - 32)
            y[l + 192] = d * scales[l // 16 + 12] * (q4 - 32)
        return y

    dequant.dequantize_q6_k = broken_q6_k
    try:
        out["catches_broken_q6"] = 0.0 if _survives(path) else 1.0
    finally:
        dequant.dequantize_q6_k = good

    return out
