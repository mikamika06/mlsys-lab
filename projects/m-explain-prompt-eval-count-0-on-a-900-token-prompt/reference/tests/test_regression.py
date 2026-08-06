import sys
sys.path.insert(0, ".")
from runner_metrics.analyzer import explain_zero_eval_count
from runner_metrics.ttft import compute_ttft


def test_counter_invariant():
    metrics = {"prompt_eval_count": 0}
    res = explain_zero_eval_count(900, metrics)
    assert isinstance(res, str)
    assert len(res) > 0


def test_ttft_positive():
    params = {"time_per_prefill_token": 0.001, "fixed_overhead": 0.02}
    val = compute_ttft(4000, params)
    assert val > 0.0
