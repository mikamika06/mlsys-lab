import pytest
from spec.ratio import recompute_draft_accept_ratio

def test_recompute_draft_accept_ratio_basic():
    data = [
        {"draft_length": 4, "accepted_count": 3},
        {"draft_length": 4, "accepted_count": 1}
    ]
    assert abs(recompute_draft_accept_ratio(data) - 0.5) < 1e-5

def test_recompute_draft_accept_ratio_zero():
    data = [{"draft_length": 0, "accepted_count": 0}]
    assert recompute_draft_accept_ratio(data) == 0.0
