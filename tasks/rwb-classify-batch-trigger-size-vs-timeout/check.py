import numpy as np

def _oracle(timestamps, batch_size, timeout):
    """
    Reference implementation used by the grader.
    Returns a 1‑D NumPy array of labels (0=size, 1=timeout).
    """
    labels = []
    n = len(timestamps)
    i = 0
    while i < n:
        start = timestamps[i]
        count = 1
        j = i + 1
        # Add items until timeout or size reached
        while j < n and (timestamps[j] - start) < timeout and count < batch_size:
            count += 1
            j += 1
        if count == batch_size:
            labels.append(0)
        else:
            # If we ran out of data, treat as size trigger
            if j == n:
                labels.append(0)
            else:
                labels.append(1)
        i = j
    return np.array(labels, dtype=int)

def grade(sol, fx) -> dict:
    """
    Generate several random test cases and compare the candidate's output
    against the oracle.  The metric is exact_match: 1.0 if all batches match,
    otherwise 0.0.
    """
    rng = np.random.default_rng(seed=42)
    ok = 1.0
    for _ in range(5):
        # Random number of timestamps
        n = rng.integers(10, 30)
        # Generate increasing timestamps with occasional large gaps
        inter_arrivals = rng.exponential(scale=0.5, size=n).cumsum()
        timestamps = inter_arrivals
        batch_size = int(rng.integers(2, 6))
        timeout = rng.uniform(1.0, 3.0)
        try:
            got = sol.classify_batches(timestamps, batch_size, timeout)
            got = np.asarray(got, dtype=int).ravel()
            ref = _oracle(timestamps, batch_size, timeout)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
