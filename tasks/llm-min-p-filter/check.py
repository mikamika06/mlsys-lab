import numpy as np

def grade(sol, fx) -> dict:
    # Reference implementation used by the grader
    def reference(probs, min_p):
        max_prob = probs.max()
        threshold = min_p * max_prob
        return probs >= threshold

    try:
        # Test with a few random probability vectors
        rng = np.random.default_rng(42)
        ok = 1.0
        for _ in range(5):
            size = rng.integers(2, 20)
            probs = rng.random(size)
            min_p = rng.random() * 0.9 + 0.1  # between 0.1 and 1.0
            expected = reference(probs, min_p)
            got = sol.minp_filter(probs, min_p)
            if not isinstance(got, np.ndarray) or got.shape != probs.shape:
                ok = 0.0
                break
            if not np.array_equal(got, expected):
                ok = 0.0
                break
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
