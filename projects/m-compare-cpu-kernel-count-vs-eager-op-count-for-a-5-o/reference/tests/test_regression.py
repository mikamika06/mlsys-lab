from audit.core import analyze_pointwise_chain, find_autotuned_config
import numpy as np


def test_pointwise_analysis():
    trace = {"eager_ops": 5, "cpu_kernels": 1}
    res = analyze_pointwise_chain(trace)
    assert res["eager_ops"] == 5
    assert res["cpu_kernels"] == 1
    assert res["ratio"] == 0.2


def test_autotuned_config_matching():
    arr = np.zeros((4, 4), dtype=np.float32)
    candidates = [{"config_id": 0, "output": arr + 1.0}, {"config_id": 1, "output": arr}]
    best = find_autotuned_config(arr, candidates)
    assert best == 1
