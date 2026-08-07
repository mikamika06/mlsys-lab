import sys
sys.path.insert(0, ".")
from flopcalc.mfu import calculate_mfu

def test_mfu_scaling_factor():
    tokens = 2000.0
    params = 1000000.0
    peak = 1e12
    mfu = calculate_mfu(tokens, params, peak)
    expected = (tokens * 6.0 * params) / peak
    assert abs(mfu - expected) < 1e-6, f"MFU scale factor incorrect: got {mfu}, expected {expected}"

def test_mfu_bounds():
    mfu = calculate_mfu(100.0, 1000.0, 1e15)
    assert 0.0 <= mfu <= 1.0, f"MFU out of reasonable bounds: {mfu}"
