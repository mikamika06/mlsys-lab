from tax.overhead import compute_pair_tax
from tax.profiles import calculate_expected_acceptance
from tax.reports import find_optimal_gamma


def test_acceptance_monotonicity():
    probs = [0.9, 0.8, 0.7]
    res = calculate_expected_acceptance(probs, 3)
    assert res["expected_accepted"] > 1.0
    assert len(res["cum_probs"]) == 3
    assert res["cum_probs"][0] >= res["cum_probs"][1] >= res["cum_probs"][2]


def test_overhead_tax_math():
    pair = {
        "pair_id": "test-pair",
        "draft_step_ms": 2.0,
        "target_step_ms": 20.0,
        "verify_step_ms": {1: 20.2, 2: 20.5, 3: 21.0},
        "acceptance_probs": [0.9, 0.8, 0.7],
    }
    res = compute_pair_tax(pair, 2)
    assert "overhead_tax" in res
    assert "speedup" in res
    assert res["speedup"] > 0
    eff = res["effective_latency_ms"]
    tax = (eff - pair["target_step_ms"]) / pair["target_step_ms"]
    assert abs(res["overhead_tax"] - tax) < 1e-7


def test_optimal_gamma_selection():
    pair = {
        "pair_id": "test-pair",
        "draft_step_ms": 10.0,
        "target_step_ms": 12.0,
        "verify_step_ms": {1: 12.5, 2: 15.0, 3: 20.0},
        "acceptance_probs": [0.1, 0.05, 0.01],
    }
    opt = find_optimal_gamma(pair, 3)
    assert opt["gamma"] == 1
