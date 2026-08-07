import sys
sys.path.insert(0, ".")
from specbench.measure import compute_acceptance_rate
from specbench.metrics import family_gap_ratio
from specbench.policy import select_viable_pairing


def test_compute_acceptance_rate_bounds():
    draft = [1, 2, 3, 4]
    target = [1, 2, 9, 4]
    probs = [{"target_prob": 0.9, "draft_prob": 0.5}, {"target_prob": 0.8, "draft_prob": 0.8}, {"target_prob": 0.1, "draft_prob": 0.5}, {"target_prob": 0.5, "draft_prob": 0.5}]
    rate = compute_acceptance_rate(draft, target, probs)
    assert 0.0 <= rate <= 1.0


def test_family_gap_ratio_calculation():
    same = [0.8, 0.85, 0.9]
    cross = [0.4, 0.45, 0.5]
    ratio = family_gap_ratio(same, cross)
    assert ratio > 1.0


def test_select_viable_pairing_filters():
    pairings = [{"name": "same-fam", "acceptance_rate": 0.85}, {"name": "cross-fam", "acceptance_rate": 0.35}]
    viable = select_viable_pairing(pairings, 0.5)
    assert "same-fam" in viable
    assert "cross-fam" not in viable
