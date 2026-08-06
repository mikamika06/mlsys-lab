import sys
sys.path.insert(0, ".")
from specadapt.tuning import estimate_alpha, adaptive_gamma

def test_estimate_alpha_bounds():
    val = estimate_alpha([0.2, 0.3, 0.4])
    assert 0.0 <= val <= 1.0

def test_adaptive_gamma_limits():
    g1 = adaptive_gamma(0.9, 8)
    assert 1 <= g1 <= 8
    g2 = adaptive_gamma(0.1, 1)
    assert 1 <= g2 <= 8
