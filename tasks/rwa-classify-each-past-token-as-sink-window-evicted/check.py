import numpy as np

def _reference(k, w, pos):
    """
    Compute the expected label array for given k, w, pos.
    Labels: 0 = sink, 1 = window, 2 = evicted.
    """
    labels = np.full(pos, 2, dtype=np.int64)          # start with evicted
    if pos == 0:
        return labels
    win_start = max(0, pos - w)
    labels[win_start:pos] = 1                         # window
    if k > 0:
        labels[:k] = 0                                 # sink overrides window
    return labels

def grade(sol, fx) -> dict:
    """
    Grade the student's implementation against a NumPy oracle.
    Returns a dictionary with metric 'exact_match'.
    """
    test_cases = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 3, 7),
        (5, 2, 10),
        (3, 4, 6),
        (0, 5, 8),
        (4, 0, 9),
    ]
    ok = 1.0
    for k, w, pos in test_cases:
        try:
            got = sol.classify_past_tokens(k, w, pos)
            ref = _reference(k, w, pos)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray):
            return {"exact_match": 0.0}
        if got.shape != ref.shape or not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
