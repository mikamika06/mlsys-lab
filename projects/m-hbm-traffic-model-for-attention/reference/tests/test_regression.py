from attention.roofline import classify_roofline


def test_classify_ridge_point():
    assert classify_roofline(10.0, 100.0, 10.0) == "compute_bound"
    assert classify_roofline(9.9, 100.0, 10.0) == "memory_bound"


def test_extreme_intensity():
    assert classify_roofline(1000.0, 100.0, 10.0) == "compute_bound"
    assert classify_roofline(0.1, 100.0, 10.0) == "memory_bound"
