import sys
sys.path.insert(0, ".")
from serving.queue import simulate


def test_simulator_uses_max_seq_len():
    res = simulate([0.0, 0.0], [10, 100], 2, 50.0)
    assert abs(res["p50"] - 31.0) < 1e-5
    assert abs(res["p99"] - 31.0) < 1e-5
