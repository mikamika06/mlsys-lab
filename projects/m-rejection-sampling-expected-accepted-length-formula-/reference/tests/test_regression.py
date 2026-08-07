import sys
sys.path.insert(0, ".")
from specdec.analysis import expected_accepted_length
from specdec.trace import compute_acceptance_drop
from specdec.profile import speculative_speedup

def test_expected_accepted_length_bounds():
    probs = [0.8, 0.8, 0.8, 0.8]
    val = expected_accepted_length(probs)
    assert val > 0.0
    assert val <= len(probs)

def test_compute_acceptance_drop_positive():
    trace = {"short_context": [0.9, 0.9, 0.9], "long_context": [0.5, 0.5, 0.5]}
    drop = compute_acceptance_drop(trace)
    assert drop > 0.0

def test_speculative_speedup_values():
    short_m = {"target_time_per_token": 10.0, "draft_time_per_token": 1.0, "gamma": 4, "acceptance_rate": 0.8}
    long_m = {"target_time_per_token": 50.0, "draft_time_per_token": 2.0, "gamma": 4, "acceptance_rate": 0.3}
    res = speculative_speedup(short_m, long_m)
    assert res["speedup_short"] > res["speedup_long"]
