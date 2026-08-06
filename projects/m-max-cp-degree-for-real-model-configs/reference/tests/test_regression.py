import sys

sys.path.insert(0, ".")
from cpdegree.config import max_cp_degree
from cpdegree.usp import hybrid_usp_bandwidth
from cpdegree.logeval import evaluate_throughput

CONFIG = {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}


def test_max_cp_degree_bounds():
    deg = max_cp_degree(CONFIG)
    assert deg <= CONFIG["num_key_value_heads"]
    assert deg > 0


def test_usp_bandwidth_positive():
    bw = hybrid_usp_bandwidth(CONFIG, 900.0, 50.0)
    assert bw > 0.0


def test_throughput_decreases_with_degree():
    t1 = evaluate_throughput(CONFIG, 1, 1000.0)
    t2 = evaluate_throughput(CONFIG, 4, 1000.0)
    assert t1 > t2
