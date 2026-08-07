import sys
sys.path.insert(0, ".")
from quantcal.limits import max_workable_samples

def test_max_workable_samples_strict_budget():
    s = max_workable_samples(None, None, 0)
    assert s == 0
    s2 = max_workable_samples(None, None, 5000)
    assert s2 <= 100
