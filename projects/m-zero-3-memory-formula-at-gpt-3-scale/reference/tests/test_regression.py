import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from zero3.schedule import build_schedule

def test_compute_requires_active_memory():
    sched = build_schedule(4, 1)
    active = set()
    for op, i in sched:
        if op in ("all_gather_fw", "all_gather_bw"):
            active.add(i)
        elif op in ("free_fw", "free_bw"):
            active.discard(i)
        elif op in ("compute_fw", "compute_bw", "reduce_scatter"):
            assert i in active, f"Layer {i} is not active during {op}"
