import numpy as np
from typing import Dict

def _reference(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "nearest":
        return arr.astype(np.float32)
    elif mode == "trunc":
        return np.trunc(arr).astype(np.float32)
    else:
        raise ValueError("mode must be 'nearest' or 'trunc'")

def grade(sol, fx) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    for _ in range(10):
        size = rng.integers(5, 50)
        arr = rng.standard_normal(size).astype(np.float64) * 1000
        for mode in ("nearest", "trunc"):
            try:
                got = sol.predict_rounding_results(arr, mode)
                ref = _reference(arr, mode)
            except Exception:
                return {"exact_match": 0.0}
            if not np.array_equal(got, ref):
                return {"exact_match": 0.0}
    return {"exact_match": 1.0}
