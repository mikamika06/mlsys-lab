import sys
sys.path.insert(0, ".")
from admit.queue import OverloadQueue
from admit.policy import estimate_latency, should_admit

def test_queue_capacity():
    q = OverloadQueue(2)
    assert q.push("a", 1) is True
    assert q.push("b", 1) is True
    assert q.push("c", 1) is False
    assert q.size() == 2

def test_priority_order():
    q = OverloadQueue(3)
    q.push("low", 1)
    q.push("high", 10)
    q.push("mid", 5)
    assert q.pop() == "high"
    assert q.pop() == "mid"
    assert q.pop() == "low"

def test_admission_policy():
    lat = estimate_latency(5, 2.0)
    assert lat == 2.5
    assert should_admit(2.5, 3.0, 0) is True
    assert should_admit(3.5, 3.0, 0) is False
