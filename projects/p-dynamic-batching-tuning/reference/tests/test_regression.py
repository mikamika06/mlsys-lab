import sys
sys.path.insert(0, ".")
from batching.profiler import measure_latency_curve
from batching.window import find_optimal_window
from batching.queues import TieredQueueManager

def test_latency_monotonicity():
    curve = measure_latency_curve([1, 2, 4, 8, 16])
    prev = 0
    for b in sorted(curve.keys()):
        assert curve[b] >= prev
        prev = curve[b]

def test_window_selection():
    curve = {1: 15.0, 4: 25.0, 8: 40.0}
    res = find_optimal_window(curve, 50.0)
    assert res["optimal_window"] > 0.0

def test_tiered_queues():
    qm = TieredQueueManager([10, 50])
    qm.push({"id": 1, "size": 5})
    qm.push({"id": 2, "size": 100})
    b = qm.pop_batch(2)
    assert len(b) == 2
