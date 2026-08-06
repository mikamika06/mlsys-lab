import sys
sys.path.insert(0, ".")
from memprof.timeline import extract_peak_allocated_bytes
from memprof.oom import find_largest_live_allocation
from memprof.fragmentation import simulate_allocator_fragmentation

def test_peak_allocated_positive():
    data = {"events": [{"action": "alloc", "size": 100}, {"action": "free", "size": 50}]}
    assert extract_peak_allocated_bytes(data) == 100

def test_largest_live_allocation():
    snapshot = {"allocations": [{"size": 200, "status": "live"}, {"size": 500, "status": "dead"}]}
    res = find_largest_live_allocation(snapshot)
    assert res["size"] == 200

def test_fragmentation_bounds():
    ops = {"total_memory": 1000, "ops": [{"type": "alloc", "size": 400}]}
    frag = simulate_allocator_fragmentation(ops)
    assert 0.0 <= frag <= 1.0
