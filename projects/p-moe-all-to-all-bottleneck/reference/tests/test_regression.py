import sys
sys.path.insert(0, ".")
from moe_opt.model import measure_communication, analyze_imbalance, group_tokens, optimized_moe_step
import numpy as np

def test_communication_positive():
    tokens = np.ones((10, 4))
    rmap = np.eye(10, 4)
    assert measure_communication(tokens, rmap) >= 0

def test_imbalance_bound():
    tokens = np.ones((10, 4))
    rmap = np.ones((10, 4))
    imb = analyze_imbalance(tokens, rmap)
    assert imb > 0

def test_group_tokens_shape():
    tokens = np.arange(20).reshape(5, 4)
    rmap = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]
    ])
    g = group_tokens(tokens, rmap)
    assert len(g) == 2
    assert sum(len(x) for x in g) == 5

def test_optimized_step():
    tokens = np.ones((8, 4))
    rmap = np.zeros((8, 4))
    rmap[0:4, 0] = 1
    rmap[4:8, 1] = 1
    res = optimized_moe_step(tokens, rmap, lambda x: x * 2)
    assert len(res) == 4
