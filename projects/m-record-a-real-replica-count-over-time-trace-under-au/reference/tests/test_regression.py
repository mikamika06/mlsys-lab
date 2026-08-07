import sys
sys.path.insert(0, ".")
from servetrace.diagnose import detect_thrashing

def test_detect_thrashing_identifies_oscillations():
    trace = [(0, 1), (1, 5), (2, 1), (3, 5), (4, 1), (5, 5)]
    assert detect_thrashing(trace, threshold=3) is True

def test_detect_thrashing_ignores_stable_trace():
    trace = [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3), (5, 3)]
    assert detect_thrashing(trace, threshold=3) is False
