import sys
sys.path.insert(0, ".")
from flashwrap.timing import measure_backward_time


def test_packed_vs_unpacked_timing_types():
    res_packed = measure_backward_time(True, None, None, None)
    res_unpacked = measure_backward_time(False, "q", "k", "v")
    assert res_packed["type"] == "packed"
    assert res_unpacked["type"] == "unpacked"
    assert isinstance(res_packed["time"], float)
    assert isinstance(res_unpacked["time"], float)
