import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    ok = 1.0
    for _ in range(5):
        size = rng.integers(3, 10)
        bis = rng.uniform(0, 1, size=size).astype(np.float64)
        threshold = rng.uniform(0, 1)
        try:
            got = sol.classify_removable_layers(bis, threshold)
            got_set = set(got)
        except Exception:
            ok = 0.0
            break
        ref_set = set(np.where(bis < threshold)[0])
        if got_set != ref_set:
            ok = 0.0
            break
    return {"exact_match": ok}
