import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from specbatch.measure import measure_tokens_per_sec
from specbatch.flops import flops_neutral_batch_size
from specbatch.crossover import find_crossover_batch_size

def test_measure_batch_scaling():
    trace = [(4, 10, 1.0)]
    assert abs(measure_tokens_per_sec(trace, 1) - 10.0) < 1e-5
    assert abs(measure_tokens_per_sec(trace, 10) - 1.0) < 1e-5

def test_flops_neutral_logic():
    assert flops_neutral_batch_size(100, 0.1, 4) == 71

def test_crossover_logic():
    sweep = {1: (10.0, 20.0), 2: (25.0, 20.0)}
    assert find_crossover_batch_size(sweep) == 1

    sweep2 = {1: (10.0, 5.0), 2: (25.0, 10.0)}
    assert find_crossover_batch_size(sweep2) == 0
