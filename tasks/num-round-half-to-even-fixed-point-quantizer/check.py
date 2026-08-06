import numpy as np

def _expected(arr, frac_bits):
    """Reference implementation using NumPy's round-half-to-even."""
    scaled = arr.astype(np.float64) * (1 << frac_bits)
    return np.round(scaled, decimals=0).astype(np.int64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok = 1.0

    # Random shapes and fractional bits
    for shape in [(10,), (5, 3), (4, 4, 4)]:
        arr = rng.uniform(-10, 10, size=shape).astype(np.float64)
        frac_bits = rng.integers(0, 8)   # 0–7 fractional bits
        try:
            got = sol.quantize_fixed_point(arr.tolist(), frac_bits)
        except Exception:
            return {"exact_match": 0.0}
        exp = _expected(arr, frac_bits)
        if got != exp.tolist():
            ok = 0.0
            break

    # Boundary case: values that are exactly .5 after scaling
    arr = np.array([0.125, 0.375], dtype=np.float64)   # *4 => [0.5, 1.5]
    frac_bits = 2
    try:
        got = sol.quantize_fixed_point(arr.tolist(), frac_bits)
    except Exception:
        return {"exact_match": 0.0}
    exp = _expected(arr, frac_bits)
    if got != exp.tolist():
        ok = 0.0

    return {"exact_match": ok}
