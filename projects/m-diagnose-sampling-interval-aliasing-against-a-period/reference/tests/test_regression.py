import sys

sys.path.insert(0, ".")
from sampler.core import simulate_execution, statistical_sample, ground_truth_fraction
from sampler.diagnose import compute_bias


def test_ground_truth_uniform():
    trace = simulate_execution(100, 10, 3)
    gt = ground_truth_fraction(trace)
    assert 0.29 <= gt <= 0.31


def test_aliasing_detection():
    biases = compute_bias(10, 3, [10, 20, 31], steps=1000)
    assert len(biases) == 3
    assert abs(biases[0]["bias"]) > 0.1 or abs(biases[1]["bias"]) > 0.1
