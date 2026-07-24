import numpy as np

def _reference(errors, k):
    # Stable tie‑break: lower index first when errors equal.
    order = np.lexsort((np.arange(len(errors)), -errors))
    return list(order[:k])

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(5):
        n = np.random.randint(5, 20)
        k = np.random.randint(0, n + 1)
        errors = np.abs(np.random.randn(n))  # positive errors
        try:
            got = sol.select_fp16_layers(errors, k)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, (list, tuple)):
            return {"exact_match": 0.0}
        ref = _reference(errors, k)
        if list(got) != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
