import numpy as np
from mlsys import scorers

def _make_test_data() -> np.ndarray:
    """Deterministic array of float32 values covering normal, special, and edge cases."""
    return np.array([
         1.0,    -1.0,
         0.0,    -0.0,
         np.float32('inf'),  np.float32('-inf'),
         3.14,   -2.71,
         1e-38,  -1e-38,
         np.finfo(np.float32).max,  np.finfo(np.float32).min,
         np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
         0.5,    -0.5,
         42.0,   -42.0,
         100.0,  -100.0,
    ], dtype=np.float32)

def _oracle_ref(arr: np.ndarray) -> np.ndarray:
    """NumPy oracle: extract sign bit via uint32 view + shift."""
    return (arr.view(np.uint32) >> 31).astype(np.uint8)

def grade(sol, fx) -> dict:
    arr = _make_test_data()
    ref = _oracle_ref(arr)

    try:
        got = sol.extract_sign_bit(arr.copy())
    except Exception:
        return {"byte_exact_fraction": 0.0}

    got = np.asarray(got, dtype=np.uint8)

    if got.shape != ref.shape:
        return {"byte_exact_fraction": 0.0}

    score = scorers.byte_exact_fraction(ref, got)
    return {"byte_exact_fraction": score}
