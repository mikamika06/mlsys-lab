import numpy as np


def _oracle(x, y):
    # The intermediate is the result of the real NumPy operation, so its dtype
    # and allocation size come from NumPy rather than a hand-written constant.
    z = np.multiply(x, y)
    out = np.add(z, 1)
    unfused = x.nbytes + y.nbytes + z.nbytes + z.nbytes + out.nbytes
    fused = x.nbytes + y.nbytes + out.nbytes
    return unfused / fused


def grade(sol, fx) -> dict:
    cases = [
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((100, 37), dtype=np.float64),
        np.zeros((7, 13, 5), dtype=np.int16),
        np.zeros((3, 257), dtype=np.float32),
    ]
    ok = 1.0
    for x in cases:
        y = np.ones_like(x)
        try:
            got = float(sol.fused_elementwise_speedup(x, y))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(x, y)
        if not np.isclose(got, ref, rtol=1e-9, atol=1e-12):
            ok = 0.0
            break
    return {"size_ratio": ok}
