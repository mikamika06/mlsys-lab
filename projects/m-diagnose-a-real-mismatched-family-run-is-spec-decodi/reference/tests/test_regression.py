import sys
sys.path.insert(0, ".")
from specdiag.metrics import compute_acceptance_metrics
from specdiag.diagnose import diagnose_run

def test_acceptance_non_negative():
    cfg = {"gamma": 3, "t_draft": 1.0, "t_target": 5.0, "acceptance_probs": [0.5, 0.4, 0.3]}
    res = compute_acceptance_metrics(cfg)
    assert res["expected_accepted"] >= 0.0

def test_mismatched_family_detection():
    cfg = {"gamma": 4, "t_draft": 3.0, "t_target": 4.0, "acceptance_probs": [0.1, 0.05, 0.01, 0.0]}
    res = diagnose_run(cfg)
    assert res["net_helping"] is False

def test_speedup_scaling():
    cfg = {"gamma": 2, "t_draft": 0.5, "t_target": 10.0, "acceptance_probs": [0.9, 0.9]}
    res = diagnose_run(cfg)
    assert res["speedup"] > 1.0
