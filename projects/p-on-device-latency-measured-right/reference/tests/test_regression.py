import sys
sys.path.insert(0, ".")
from edgelat.profiler import LatencyProfiler


def test_cold_start_measurement():
    p = LatencyProfiler([42.0, 10.0, 10.0])
    assert p.measure_cold_start() == 42.0


def test_steady_state_separation():
    p = LatencyProfiler([50.0, 10.0, 10.0, 10.0])
    first, steady = p.separate_first_and_steady()
    assert first == 50.0
    assert steady == 10.0
