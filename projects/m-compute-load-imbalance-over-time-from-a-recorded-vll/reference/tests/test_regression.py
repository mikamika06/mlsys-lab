import numpy as np
from moe_analyzer.imbalance import compute_imbalance_over_time


def test_imbalance_detection():
    log_entries = [
        {
            "timestamp": 100.0,
            "raw_expert_tokens": [100, 200, 100, 400],
            "eplb_expert_tokens": [200, 200, 200, 200],
        },
        {
            "timestamp": 101.0,
            "raw_expert_tokens": [300, 300, 300, 300],
            "eplb_expert_tokens": [300, 300, 300, 300],
        },
    ]
    res = compute_imbalance_over_time(log_entries)
    assert len(res["imbalance_ratios"]) == 2
    assert np.isclose(res["imbalance_ratios"][0], 2.0)
    assert np.isclose(res["eplb_effective_ratios"][0], 1.0)
    assert np.isclose(res["imbalance_ratios"][1], 1.0)
