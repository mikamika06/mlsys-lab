import sys

sys.path.insert(0, ".")
from speculative.metrics import compute_acceptance_rate
from speculative.eval import compare_families, classify_pairing
from speculative.pairing import select_best_draft


def test_acceptance_rate_bounds():
    trace = {"draft_tokens": [1, 2, 3, 4], "accepted_tokens": [1, 2]}
    rate = compute_acceptance_rate(trace)
    assert 0.0 <= rate <= 1.0, f"acceptance rate {rate} out of bounds"


def test_empty_trace_rate():
    trace = {"draft_tokens": [], "accepted_tokens": []}
    rate = compute_acceptance_rate(trace)
    assert rate == 0.0, f"expected 0.0 for empty trace, got {rate}"


def test_same_vs_cross_comparison():
    same = {"draft_tokens": [1, 2, 3], "accepted_tokens": [1, 2, 3]}
    cross = {"draft_tokens": [1, 2, 3], "accepted_tokens": [1]}
    diff = compare_families(same, cross)
    assert diff > 0.0, f"expected positive diff for same vs cross, got {diff}"


def test_select_best_draft_validity():
    cands = [
        {"draft_tokens": [1, 2], "accepted_tokens": [1]},
        {"draft_tokens": [1, 2], "accepted_tokens": [1, 2]}
    ]
    best = select_best_draft(cands)
    assert best == 1, f"expected index 1, got {best}"
