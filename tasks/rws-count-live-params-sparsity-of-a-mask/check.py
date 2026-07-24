def grade(sol, fx) -> dict:
    import numpy as np

    # Define a set of test masks
    cases = [
        np.array([[0, 1], [1, 0]]),
        np.random.randint(2, size=(5, 7)),
        np.zeros((10, 3), dtype=int),
        np.ones((4, 4), dtype=int),
        np.random.randint(2, size=(20, 15))
    ]

    ok = 1.0
    for mask in cases:
        try:
            got_live, got_sparsity = sol.count_live_params_and_sparsity(mask)
        except Exception:
            ok = 0.0
            break

        expected_live = int(np.count_nonzero(mask))
        expected_sparsity = float((mask.size - expected_live) / mask.size)

        if got_live != expected_live or abs(got_sparsity - expected_sparsity) > 1e-12:
            ok = 0.0
            break

    return {"exact_match": ok}
