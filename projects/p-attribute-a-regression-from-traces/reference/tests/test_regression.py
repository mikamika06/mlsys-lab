import sys
sys.path.insert(0, ".")
from profiler.comparator import TraceComparator
import ref

def test_comparator_basic():
    ta, tb, _ = ref.generate_traces()
    comp = TraceComparator(ta, tb)
    table = comp.reduce_trace(ta)
    assert len(table) > 0

def test_max_delta():
    ta, tb, _ = ref.generate_traces()
    comp = TraceComparator(ta, tb)
    _, delta = comp.find_max_delta()
    assert delta >= 0.0
