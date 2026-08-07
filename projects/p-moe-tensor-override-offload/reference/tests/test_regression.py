import sys
sys.path.insert(0, ".")
from moe_offload.offload import MoEOffloader
import numpy as np

def test_frequency_measurement():
    sizes = [100, 100]
    traces = [[0], [0], [1]]
    off = MoEOffloader(sizes, 200)
    freqs = off.measure_frequencies(traces)
    assert np.allclose(freqs, [2/3, 1/3])

def test_compute_rules():
    sizes = [100, 500]
    off = MoEOffloader(sizes, 200)
    freqs = [0.9, 0.1]
    rules = off.compute_rules(freqs, 200)
    assert 1 in rules

def test_latency_evaluation():
    sizes = [100, 100]
    off = MoEOffloader(sizes, 200)
    lat = off.evaluate_latency({0}, [10.0, 10.0], penalty_factor=2.0)
    assert lat == 30.0

def test_verify_output():
    sizes = [100]
    off = MoEOffloader(sizes, 100)
    assert off.verify_output([1.0], [1.0000001])

def test_check_constraints():
    sizes = [100, 100]
    off = MoEOffloader(sizes, 150)
    assert off.check_constraints({1}, 150, 20.0, 30.0)
