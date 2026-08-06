import sys
sys.path.insert(0, ".")
from itl_debug.metrics import detect_period
from itl_debug.fix import apply_fix


def test_fix_removes_periodic_spikes():
    raw = [10.0] * 100
    for i in range(10, 100, 10):
        raw[i] = 60.0
    fixed = apply_fix(raw)
    assert max(fixed) < 30.0, "periodic spikes still present"


def test_period_detection():
    raw = [10.0] * 200
    for i in range(25, 200, 25):
        raw[i] = 70.0
    assert detect_period(raw) == 25
