import sys

sys.path.insert(0, ".")
from specdec.metrics import expected_accepted_tokens, expected_total_tokens, measure_acceptance_rates
from specdec.tuning import evaluate_draft_throughput, select_optimal_draft_max


def test_expected_tokens_monotonic_with_acceptance():
    r1 = [0.8, 0.8, 0.8]
    r2 = [0.5, 0.5, 0.5]
    assert expected_accepted_tokens(r1) > expected_accepted_tokens(r2)


def test_expected_total_tokens_includes_bonus_target_token():
    rates = [0.9, 0.8]
    exp_acc = expected_accepted_tokens(rates)
    exp_tot = expected_total_tokens(rates)
    assert abs(exp_tot - (exp_acc + 1.0)) < 1e-9


def test_draft_latency_affects_optimal_gamma():
    rates = [0.9] * 8
    tp_light = evaluate_draft_throughput(rates, draft_time=0.001, target_time=0.05)
    tp_heavy = evaluate_draft_throughput(rates, draft_time=0.04, target_time=0.02)
    assert tp_light[8] > tp_light[1]
    assert tp_heavy[8] < tp_heavy[1]


def test_measure_acceptance_rates_bounds():
    counts = [2, 2, 2, 0]
    rates = measure_acceptance_rates(counts, draft_max=4)
    assert len(rates) == 4
    assert all(0.0 <= r <= 1.0 for r in rates)
