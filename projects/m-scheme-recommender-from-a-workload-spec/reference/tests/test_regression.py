import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quantrec.bytes import bytes_per_token
from quantrec.crossover import crossover_batch_size
from quantrec.recommend import recommend_scheme


def test_bytes_per_token_accuracy():
    assert bytes_per_token(1000.0, "W8A8") == 1000.0
    assert bytes_per_token(1000.0, "W4A16") == 500.0
    assert bytes_per_token(1000.0, "FP16") == 2000.0


def test_crossover_batch_size_positive():
    val = crossover_batch_size(900.0, 312.0)
    assert val > 0.0
    assert abs(val - 173.333333) < 0.001


def test_recommend_scheme_low_batch():
    w = {"batch_size": 1, "bandwidth_gbps": 900.0, "tflops_w16": 312.0}
    assert recommend_scheme(w) == "W4A16"


def test_recommend_scheme_high_batch():
    w = {"batch_size": 256, "bandwidth_gbps": 900.0, "tflops_w16": 312.0}
    assert recommend_scheme(w) == "W8A8"
