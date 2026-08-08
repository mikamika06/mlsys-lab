import sys

sys.path.insert(0, ".")
from roofline.classify import classify_kernel, max_achievable_gflops


def test_classify_and_roofline_boundaries():
    ridge = 100.0
    assert classify_kernel(50.0, ridge) == "memory-bound"
    assert classify_kernel(100.0, ridge) == "compute-bound"
    assert classify_kernel(200.0, ridge) == "compute-bound"

    peak = 1000.0
    bw = 10.0
    assert abs(max_achievable_gflops(50.0, peak, bw) - 500.0) < 1e-5
    assert abs(max_achievable_gflops(100.0, peak, bw) - 1000.0) < 1e-5
    assert abs(max_achievable_gflops(200.0, peak, bw) - 1000.0) < 1e-5
