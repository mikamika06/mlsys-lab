from autotune.overhead import compute_overhead_reduction

def test_overhead_bounds():
    b = [{"launch_delay": 100, "driver_wait": 50}]
    a = [{"launch_delay": 20, "driver_wait": 10}]
    res = compute_overhead_reduction(b, a)
    assert 0.0 <= res <= 1.0, "overhead reduction must be within valid probability bounds"
    assert res == 0.8
