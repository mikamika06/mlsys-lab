import sys
sys.path.insert(0, ".")
from mlx_mem.instrument import instrument_loop
from mlx_mem.throughput import measure_throughput
from mlx_mem.locate import locate_ceiling

def test_instrument_monotonic_peak():
    res = instrument_loop(5)
    peaks = [r["peak"] for r in res]
    assert all(peaks[i] <= peaks[i + 1] for i in range(len(peaks) - 1))

def test_throughput_penalty_exists():
    def_t = measure_throughput(100, False)
    lim_t = measure_throughput(100, True)
    assert lim_t < def_t

def test_locate_ceiling_found():
    logs = ["0 500", "1 1200", "2 2500"]
    assert locate_ceiling(logs, 1000) == 1
