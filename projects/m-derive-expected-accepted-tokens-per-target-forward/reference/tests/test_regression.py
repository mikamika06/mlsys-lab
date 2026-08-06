import sys
sys.path.insert(0, ".")
from specdec.metrics import expected_accepted_tokens, measure_acceptance_rate
from specdec.optimizer import find_optimal_draft_max


def test_expected_tokens_bounds():
    val = expected_accepted_tokens(4, 0.5)
    assert 0.0 <= val <= 4.0


def test_acceptance_rate_range():
    traces = [(4, 2), (4, 4), (4, 0)]
    rate = measure_acceptance_rate(traces)
    assert 0.0 <= rate <= 1.0
    assert abs(rate - 0.5) < 1e-5


def test_optimal_draft_max_positive():
    opt = find_optimal_draft_max(8, 0.8, 0.2)
    assert isinstance(opt, int)
    assert 1 <= opt <= 8
