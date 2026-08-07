import sys

sys.path.insert(0, ".")
from workset.model import compute_working_set
from workset.warmup import warmup_curve


def test_working_set_non_negative():
    trace = [1, 2, 3, 1, 2]
    res = compute_working_set(trace, 0.5)
    assert res >= 0


def test_working_set_full_coverage():
    trace = [1, 2, 3]
    res = compute_working_set(trace, 1.0)
    assert res == 3


def test_warmup_curve_length():
    trace = [1, 2, 3, 4, 5]
    curve = warmup_curve(trace, 2)
    assert len(curve) == len(trace)


def test_warmup_curve_values():
    trace = [1, 1, 1]
    curve = warmup_curve(trace, 1)
    assert curve == [0, 1, 1]
