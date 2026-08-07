import sys
sys.path.insert(0, ".")
from pact.peak import compute_peak_activations
from pact.trace import memory_trace
from pact.budget import max_microbatches

def test_peak_activations_vary_by_stage():
    peaks = compute_peak_activations(4, 8)
    assert len(peaks) == 4
    assert peaks[0] > peaks[-1], "peak activations should decrease for later pipeline stages"
    assert peaks[0] == 4

def test_trace_shape_and_bounds():
    P, M = 3, 5
    trace = memory_trace(P, M)
    assert len(trace) > 0
    for row in trace:
        assert len(row) == P
        for s, val in enumerate(row):
            assert 0 <= val <= min(P - s, M)

def test_budget_calculation():
    m = max_microbatches(4, 1000, 200)
    assert m == 5
