import sys
import numpy as np

sys.path.insert(0, ".")
from moediag.router import find_zeroed_rows
from moediag.params import count_parameters


def test_router_rows_nonzero():
    weights = np.ones((4, 8), dtype=float)
    weights[2, :] = 0.0
    zeroed = find_zeroed_rows(weights)
    assert 2 in zeroed, f"failed to detect zeroed row at index 2, got {zeroed}"


def test_parameter_counts_positive():
    config = {
        "hidden_dim": 64,
        "num_experts": 4,
        "top_k": 2,
        "intermediate_dim": 128,
        "num_layers": 1,
    }
    counts = count_parameters(config)
    assert counts["total_parameters"] > counts["active_parameters"]
