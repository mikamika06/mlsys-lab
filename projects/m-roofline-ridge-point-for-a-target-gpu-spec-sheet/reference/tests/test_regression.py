import sys
sys.path.insert(0, ".")
from roofline.spec import compute_ridge_point
from roofline.classify import classify_decode
from roofline.measure import measure_crossover_ratio

def test_ridge_point_positive():
    val = compute_ridge_point(100.0, 10.0)
    assert val == 10.0

def test_classify_correctness():
    assert classify_decode(5.0, 10.0) == "memory-bound"
    assert classify_decode(15.0, 10.0) == "compute-bound"

def test_measure_ratio_bounds():
    ratio = measure_crossover_ratio(2.0, 10.0)
    assert ratio == 5.0
