import numpy as np

def _oracle(max_bucket, batch_sizes):
    """Reference implementation used by the grader."""
    return np.asarray(batch_sizes) <= max_bucket

def grade(sol, fx) -> dict:
    # Test cases covering edge conditions
    cases = [
        (5, [0, 3, 5]),          # all captured
        (10, [11, 12, 13]),      # all eager
        (7, [7, 8, 6]),          # mix with boundary
        (1, []),                 # empty input
        (100, list(range(0, 200, 20))),  # larger range
    ]
    ok = 1.0
    for max_bucket, sizes in cases:
        try:
            got = sol.classify_batches(max_bucket, sizes)
            expected = _oracle(max_bucket, sizes)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(got, expected):
            ok = 0.0
            break
    return {"exact_match": ok}
