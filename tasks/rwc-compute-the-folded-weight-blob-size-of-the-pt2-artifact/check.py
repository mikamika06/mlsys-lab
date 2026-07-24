import numpy as np

def _reference_size(params):
    total = 0
    for arr in params.values():
        total += arr.size * arr.dtype.itemsize
    return int(total)

def grade(sol, fx) -> dict:
    # Define a few representative test cases
    cases = [
        {
            "params": {
                "a": np.ones((10, 5), dtype=np.float32),
                "b": np.zeros(3, dtype=np.int64)
            }
        },
        {
            "params": {
                "x": np.arange(12, dtype=np.uint8).reshape(4, 3),
                "y": np.full((2,), 7.5, dtype=np.float16)
            }
        },
        {
            "params": {
                "p": np.random.randn(7, 7).astype(np.double),
                "q": np.zeros((1, 2, 3), dtype=np.int32)
            }
        }
    ]

    ok = 1.0
    for case in cases:
        params = case["params"]
        try:
            got = sol.compute_folded_weight_blob_size(params)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_size(params)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
