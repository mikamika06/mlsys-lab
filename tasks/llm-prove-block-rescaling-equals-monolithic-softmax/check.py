import numpy as np

def grade(sol, fx) -> dict:
    # Generate deterministic test cases
    rng = np.random.default_rng(0)
    test_cases = [
        rng.uniform(-5, 5, size=20),
        rng.uniform(-10, 10, size=50),
        rng.uniform(-1, 1, size=30),
        rng.standard_normal(size=40),
        rng.exponential(scale=2.0, size=25) - 3.0,
    ]
    block_sizes = [1, 3, 7, 13]
    max_err = 0.0
    for logits_arr in test_cases:
        # Ensure float64 dtype
        logits_arr = np.asarray(logits_arr, dtype=np.float64)
        M = np.max(logits_arr)
        ref = np.exp(logits_arr - M) / np.sum(np.exp(logits_arr - M))
        logits_list = logits_arr.tolist()
        for bs in block_sizes:
            try:
                cand = sol.block_rescale_softmax(logits_list, bs)
            except Exception:
                return {"max_abs_err": float("inf")}
            if not isinstance(cand, list) or len(cand) != len(logits_list):
                return {"max_abs_err": float("inf")}
            cand_arr = np.array(cand, dtype=np.float64)
            err = np.max(np.abs(cand_arr - ref))
            if err > max_err:
                max_err = err
    return {"max_abs_err": max_err}
