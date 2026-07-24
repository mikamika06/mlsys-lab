import numpy as np

def _oracle(X, threshold):
    """Compute migration channel indices using the same algorithm as the spec."""
    X = np.asarray(X, dtype=np.float64)
    channel_absmax = np.max(np.abs(X), axis=0)
    rho = np.median(channel_absmax)
    flagged = np.where(channel_absmax > threshold * rho)[0]
    return sorted(int(i) for i in flagged)

def _make_test_cases():
    """Return list of (X, threshold) pairs covering diverse scenarios."""
    rng = np.random.RandomState(123)
    cases = []

    # Case 1: clear outlier in channel 2
    X1 = rng.randn(32, 8).astype(np.float64)
    X1[:, 2] *= 20.0
    cases.append((X1, 3.0))

    # Case 2: no outliers — nothing should be flagged
    X2 = rng.randn(64, 16).astype(np.float64)
    cases.append((X2, 3.0))

    # Case 3: multiple outlier channels
    X3 = rng.randn(48, 12).astype(np.float64)
    X3[:, 1] *= 15.0
    X3[:, 5] *= 10.0
    X3[:, 9] *= 12.0
    cases.append((X3, 4.0))

    # Case 4: single-row matrix
    X4 = np.array([[1.0, 5.0, 0.1, 0.2]], dtype=np.float64)
    cases.append((X4, 2.0))

    # Case 5: threshold = 1.0, only median value itself should not exceed cutoff
    X5 = rng.randn(80, 32).astype(np.float64)
    X5[:, 0] = 100.0
    X5[:, 15] = -90.0
    cases.append((X5, 1.0))

    # Case 6: all channels have identical absmax
    X6 = np.ones((10, 5), dtype=np.float64)
    cases.append((X6, 3.0))

    # Case 7: negative threshold edge (very small)
    X7 = rng.randn(20, 6).astype(np.float64)
    X7[:, 3] *= 50.0
    cases.append((X7, 0.5))

    return cases

def grade(sol, fx) -> dict:
    cases = _make_test_cases()
    ok = 1.0
    for X, threshold in cases:
        try:
            got = sol.migration_channels(X, threshold)
            got = sorted(int(i) for i in got)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(X, threshold)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
