def _reference(x, y, mask):
    import numpy as np
    mins = np.minimum(x, y)
    maxs = np.maximum(x, y)
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    sel = np.where(mask, x, y)
    return (mins, maxs, abs_x, abs_y, sel)

def _byte_exact_fraction(a, b):
    if a.tobytes() == b.tobytes():
        return 1.0
    else:
        return 0.0

def grade(sol, fx) -> dict:
    import numpy as np
    rng = np.random.default_rng(12345)
    ok = 1.0
    for _ in range(10):
        shape = (rng.integers(5, 15),)
        x = rng.integers(-100, 100, size=shape, dtype=np.int32)
        y = rng.integers(-100, 100, size=shape, dtype=np.int32)
        mask = rng.integers(0, 2, size=shape).astype(bool)
        try:
            out = sol.branchless_ops(x, y, mask)
        except Exception as e:
            return {"byte_exact_fraction": 0.0}
        if len(out) != 5:
            return {"byte_exact_fraction": 0.0}
        ref = _reference(x, y, mask)
        for a, b in zip(out, ref):
            if _byte_exact_fraction(a, b) < 1.0:
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"byte_exact_fraction": ok}
