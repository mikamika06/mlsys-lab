import sys
sys.path.insert(0, ".")
from vllmsched.scheduler import simulate_scheduler, Request

def test_scheduler_completes_all_requests():
    reqs = [Request("r1", 10, 5, 0), Request("r2", 10, 5, 0)]
    out = simulate_scheduler(reqs, policy="fcfs")
    assert len(out) == 2

def test_priority_scheduler_respects_priority():
    reqs = [Request("r1", 10, 5, 0), Request("r2", 10, 5, 1)]
    out = simulate_scheduler(reqs, policy="priority")
    assert out[0]["req_id"] == "r2"
