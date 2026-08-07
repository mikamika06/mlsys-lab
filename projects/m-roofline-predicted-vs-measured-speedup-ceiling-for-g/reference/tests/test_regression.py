import sys
sys.path.insert(0, ".")
from roofline.calc import compute_intensity, roofline_speedup_ceiling, measure_speedup_error

def test_compute_intensity_basic():
    assert compute_intensity(1000, 100) == 10.0

def test_roofline_speedup_ceiling_gamma4():
    val = roofline_speedup_ceiling(5.0, 1e12, 1e11, 10.0, 1000.0, 4)
    assert val > 0.0

def test_measure_speedup_error_bounds():
    err = measure_speedup_error(4.0, 3.8)
    assert 0.0 <= err <= 1.0
