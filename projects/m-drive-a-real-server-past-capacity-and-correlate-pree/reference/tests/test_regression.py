from servermon.correlate import compute_p99_correlation

def test_correlation_bounds():
    res = compute_p99_correlation([1, 2, 3], [10, 20, 30])
    assert -1.0 <= res["correlation"] <= 1.0
    assert res["p99"] > 0
