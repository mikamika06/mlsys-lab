import sys
sys.path.insert(0, ".")
from timingcache.cache import compute_payoff
from timingcache.optimization import find_knee
from timingcache.tactics import detect_flaky

def test_payoff_positive():
    r = compute_payoff([100.0, 200.0], [10.0, 20.0])
    assert all(x > 1.0 for x in r)

def test_knee_valid_level():
    lvl = find_knee([1, 2, 3, 4, 5], [10.0, 5.0, 4.9, 4.8, 4.8], [1.0, 2.0, 10.0, 50.0, 200.0])
    assert lvl in [1, 2, 3, 4, 5]

def test_flaky_filtering():
    runs = {"t1": [1.0, 1.0, 1.0], "t2": [1.0, 5.0, 1.0]}
    stable = detect_flaky(runs)
    assert "t1" in stable
    assert "t2" not in stable
