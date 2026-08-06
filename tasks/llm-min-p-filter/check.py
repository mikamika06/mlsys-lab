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
            probs_arr = rng.random(size)
            min_p = rng.random() * 0.9 + 0.1  # between 0.1 and 1.0
            expected_arr = reference(probs_arr, min_p)

            probs_list = probs_arr.tolist()
            expected_list = expected_arr.tolist()

            got = sol.minp_filter(probs_list, min_p)
            if not isinstance(got, list) or len(got) != len(probs_list):
                ok = 0.0
                break
            if got != expected_list:
                ok = 0.0
                break
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
