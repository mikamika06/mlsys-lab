import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cachesim.simulator import simulate

def test_dirty_eviction_penalty_from_l2():
    trace = [("W", 1, 100)]
    result = simulate(trace, l1_cap=0, l2_cap=0, policy="always", write_mode="wb")

    assert result["write_penalty_ns"] == 11000
    assert result["l1_evictions"] == 1
    assert result["l2_evictions"] == 1
