import sys
sys.path.insert(0, ".")
from turnover.schedule import schedule_batch

def test_long_output_starvation():
    active = [{"id": 1, "remaining_tokens": 1000}]
    pending = [{"id": 2, "remaining_tokens": 5}, {"id": 3, "remaining_tokens": 5}]
    res = schedule_batch(active, pending, capacity=2, policy="preemptive_budget")
    ids = [r["id"] for r in res]
    assert 2 in ids and 3 in ids, f"Short requests starved by long output request: {ids}"
