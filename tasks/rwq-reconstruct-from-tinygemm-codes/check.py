import numpy as np
from mlsys import scorers


def _ref_dequantize(codes, scale, zero_point):
    return (np.asarray(codes, dtype=np.float32) * scale + zero_point).astype(np.float32)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    worst = 0.0
    cases = [
        (np.array([0, 7, 8, 15], dtype=np.uint8), 0.1, -0.75),
        (rng.integers(0, 16, size=128).astype(np.uint8), 0.05, -0.4),
        (rng.integers(0, 16, size=256).astype(np.uint8), 0.02, 0.0),
        (np.zeros(16, dtype=np.uint8), 0.1, 0.3),
        (np.full(16, 15, dtype=np.uint8), 0.1, 0.3),
        (rng.integers(0, 16, size=512).astype(np.uint8), 0.001, -0.008),
    ]
    for codes, scale, zp in cases:
        ref = _ref_dequantize(codes, scale, zp)
        try:
            got = np.asarray(sol.dequantize_uint4(codes.copy(), float(scale), float(zp)), dtype=np.float32)
        except Exception:
            return {"max_abs_err": 1e9}
        err = scorers.max_abs_err(ref, got)
        if err > worst:
            worst = err
    return {"max_abs_err": worst}
