import numpy as np

def _reference(arr, stride):
    indices = [t * stride for t in range(32)]
    segments = set(i // 32 for i in indices)
    return len(segments)

def grade(sol, fx) -> dict:
    test_cases = [
        (np.arange(128, dtype=np.float32), 1),
        (np.arange(128, dtype=np.float32), 2),
        (np.arange(128, dtype=np.float32), 3),
        (np.arange(256, dtype=np.float32), 4),
        (np.arange(512, dtype=np.float32), 7),
    ]
    ok = 1.0
    for arr, stride in test_cases:
        try:
            got = sol.count_transactions(arr, stride)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference(arr, stride)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
