import sys
sys.path.insert(0, ".")
from serving import engine

def test_latency_increases_with_batch():
    lats = engine.latency_curve([1, 16], 10.0, 2.0, 4)
    assert lats[1] > lats[0]

def test_slo_selection_valid():
    bs = [1, 2, 4, 8]
    opt = engine.find_slo_point(bs, 100.0, 10.0, 4)
    assert opt in bs

def test_burst_not_empty():
    res = engine.simulate_burst(4, [8], 4)
    assert len(res) > 0
