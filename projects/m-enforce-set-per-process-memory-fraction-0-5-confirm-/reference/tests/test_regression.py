import sys

sys.path.insert(0, ".")
from mpsmem.fraction import enforce_fraction, check_oom
from mpsmem.loop import simulate_generation, compute_fragmentation


def test_fraction_limit():
    limit = enforce_fraction(0.5, 1000)
    assert limit == 500


def test_oom_trigger():
    assert check_oom(500, 450, 200) is True
    assert check_oom(500, 200, 100) is False


def test_generation_loop_clears():
    h, clears = simulate_generation(10, 3, 100)
    assert clears > 0


def test_fragmentation_positive():
    r = compute_fragmentation([120, 240], [100, 200])
    assert all(x >= 0 for x in r)
