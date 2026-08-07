import sys
sys.path.insert(0, ".")
from profiler.analysis import find_most_expensive_phase, verify_with_second_trace

def test_most_expensive_phase_basic():
    data = [{"name": "phaseA", "start": 0, "end": 10}, {"name": "phaseB", "start": 10, "end": 40}]
    assert find_most_expensive_phase(data) == "phaseB"

def test_verify_with_second_trace_matching():
    t1 = [{"name": "phaseA", "start": 0, "end": 10}, {"name": "phaseB", "start": 10, "end": 40}]
    t2 = [{"name": "phaseA", "start": 0, "end": 5}, {"name": "phaseB", "start": 5, "end": 35}]
    assert verify_with_second_trace(t1, t2) is True
