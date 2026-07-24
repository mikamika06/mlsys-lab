import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    np.random.seed(42)
    n_dst = 16
    n_src = 32
    dst = np.random.randint(0, 100, size=n_dst, dtype=np.int32)
    src = np.random.randint(-50, 50, size=n_src, dtype=np.int32)
    idx = np.random.randint(0, n_dst, size=n_src, dtype=np.int32)
    # Ensure at least some duplicates
    idx[:5] = 0
    out = dst.copy()
    try:
        sol.scatter_add(dst, idx, src, out)
    except Exception:
        return {"covers_all": 0.0, "byte_exact": 0.0}
    expected = dst.copy()
    np.add.at(expected, idx, src)
    covers = 1.0 if (np.min(idx) >= 0 and np.max(idx) < n_dst) else 0.0
    byte_exact = scorers.byte_exact_fraction(expected, out)
    return {"covers_all": covers, "byte_exact": byte_exact}
