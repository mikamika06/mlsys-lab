import sys
sys.path.insert(0, ".")
from batcher.policy import compute_rejected_requests
from batcher.diagnose import attribute_latency

def test_queue_rejection_limit():
    rejected = compute_rejected_requests(queue_size=80, max_queue_size=100, incoming_count=30, policy="REJECT")
    assert rejected == 10, f"Expected 10 rejected requests, got {rejected}"

def test_attribute_latency_batcher():
    res = attribute_latency(10.0, 20.0, 50.0, 22.0)
    assert res == "batcher"
