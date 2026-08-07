import sys
sys.path.insert(0, ".")
from vllm_diag.eviction import simulate_eviction


def test_eviction_under_pressure():
    ops = [
        ("access", 1, 1),
        ("access", 2, 1),
        ("release", 1, 1),
        ("access", 3, 1),
        ("access", 4, 1),
    ]
    evicted = simulate_eviction(2, ops)
    assert evicted > 0, "expected blocks to be evicted under memory pressure"
