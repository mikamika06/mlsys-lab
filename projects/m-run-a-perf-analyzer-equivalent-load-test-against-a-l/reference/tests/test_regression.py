import sys
sys.path.insert(0, ".")
from loadtest.simulator import generate_schedule
from loadtest.metrics import compute_metrics

def test_schedule_length():
    sched = generate_schedule(4, 50)
    assert len(sched) == 50

def test_metrics_accuracy():
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    m = compute_metrics(lats, 1.0)
    assert m["p50"] == 30.0
    assert m["throughput"] == 5.0
