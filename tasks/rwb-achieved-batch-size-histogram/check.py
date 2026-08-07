import numpy as np

def _ref(arrivals, batch_timeout, max_batch_size):
    """Reference implementation: scan arrivals once, dispatch on size or timeout."""
    histogram = [0] * (max_batch_size + 1)
    batch_start = None
    batch_size = 0
    for t in arrivals:
        if batch_size == 0:
            batch_start = t
            batch_size = 1
        else:
            batch_size += 1
        if batch_size == max_batch_size or (t - batch_start) >= batch_timeout:
            histogram[batch_size] += 1
            batch_size = 0
            batch_start = None
    if batch_size > 0:
        histogram[batch_size] += 1
    return histogram

def _generate_test_case():
    """Generate a deterministic arrivals trace for grading."""
    rng = np.random.default_rng(42)
    intervals = rng.exponential(scale=0.5, size=500)
    arrivals = np.cumsum(intervals)
    arrivals = arrivals[arrivals <= 300][:300]
    return arrivals.tolist()

def grade(sol, fx) -> dict:
    """Grade the batch_size_histogram implementation."""
    arrivals = _generate_test_case()
    batch_timeout = 1.5
    max_batch_size = 8

    try:
        hist = sol.batch_size_histogram(arrivals, batch_timeout, max_batch_size)
    except Exception:
        return {'exact_match': 0.0}

    ref = _ref(arrivals, batch_timeout, max_batch_size)

    # Validate output type and length
    if not isinstance(hist, list):
        return {'exact_match': 0.0}
    if len(hist) != len(ref):
        return {'exact_match': 0.0}
    if not all(isinstance(x, int) for x in hist):
        return {'exact_match': 0.0}

    # Check exact match
    if hist != ref:
        return {'exact_match': 0.0}

    return {'exact_match': 1.0}
