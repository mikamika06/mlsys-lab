import math
from typing import Dict

def _reference(arr: list[float], mode: str) -> list[float]:
    if mode == "nearest":
        return [float(val) for val in arr]
    elif mode == "trunc":
        res = []
        for val in arr:
            val_f = float(val)
            res.append(math.copysign(float(math.trunc(val_f)), val_f))
        return res
    else:
        raise ValueError("mode must be 'nearest' or 'trunc'")

def grade(sol, fx) -> Dict[str, float]:
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(10):
        size = rng.integers(5, 50)
        arr_np = rng.standard_normal(size).astype(float) * 1000
        arr = arr_np.tolist()
        for mode in ("nearest", "trunc"):
            try:
                got = sol.predict_rounding_results(arr, mode)
                ref = _reference(arr, mode)
            except Exception:
                return {"exact_match": 0.0}
            if got != ref:
                return {"exact_match": 0.0}
    return {"exact_match": 1.0}
