import sys

sys.path.insert(0, ".")
from pte.plan import replan_buffers


def test_replan_no_overlap():
    activations = [
        {"id": 1, "size": 100, "constant": False, "start": 0, "end": 2},
        {"id": 2, "size": 100, "constant": False, "start": 1, "end": 3},
    ]
    allocs = replan_buffers(activations)
    o1 = allocs[1]
    o2 = allocs[2]
    overlap = not (o1 + 100 <= o2 or o2 + 100 <= o1)
    assert not overlap
