import sys
sys.path.insert(0, ".")
from det.profiler import measure_decode_throughput

def test_throughput_invariant_to_temperature():
    t1 = measure_decode_throughput(0.0)
    t2 = measure_decode_throughput(1.0)
    assert abs(t1 - t2) < 1e-5, f"throughput changed with temperature: {t1} vs {t2}"
