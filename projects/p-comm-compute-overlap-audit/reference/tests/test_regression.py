import sys
sys.path.insert(0, ".")
from overlap.audit import compute_overlap_ratio, find_barriers

def test_overlap_ratio_bound():
    events = [
        {"name": "comp1", "start": 0, "dur": 100, "type": "compute"},
        {"name": "comm1", "start": 50, "dur": 60, "type": "comm"}
    ]
    ratio = compute_overlap_ratio(events)
    assert ratio == 0.0

def test_barriers_detection():
    events = [
        {"name": "comp1", "start": 0, "dur": 100, "type": "compute"},
        {"name": "comm1", "start": 120, "dur": 50, "type": "comm"}
    ]
    barriers = find_barriers(events)
    assert len(barriers) == 1
