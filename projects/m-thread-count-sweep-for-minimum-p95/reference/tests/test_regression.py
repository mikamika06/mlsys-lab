import sys
import numpy as np

sys.path.insert(0, ".")
from threadsweep.sla import compute_sla_throughput


def test_throughput_non_negative():
    latencies = [np.array([10.0, 12.0, 15.0]), np.array([20.0, 22.0, 25.0])]
    tp = compute_sla_throughput(latencies, 30.0)
    assert tp >= 0.0


def test_throughput_zero_when_sla_impossible():
    latencies = [np.array([100.0, 120.0, 150.0])]
    tp = compute_sla_throughput(latencies, 10.0)
    assert tp == 0.0
