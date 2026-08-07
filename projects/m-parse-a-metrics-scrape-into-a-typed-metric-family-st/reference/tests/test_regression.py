import sys

sys.path.insert(0, ".")
from vllm_metrics.rates import compute_counter_rates


def test_counter_reset_handled_correctly():
    f1 = {
        "name": "vllm:num_requests_total",
        "type": "counter",
        "samples": [
            {
                "name": "vllm:num_requests_total",
                "labels": {"engine": "0"},
                "value": 1000.0,
            }
        ],
    }
    f2 = {
        "name": "vllm:num_requests_total",
        "type": "counter",
        "samples": [
            {
                "name": "vllm:num_requests_total",
                "labels": {"engine": "0"},
                "value": 20.0,
            }
        ],
    }
    rates = compute_counter_rates(f1, f2, 10.0)
    key = frozenset([("engine", "0")])
    assert key in rates
    assert rates[key] == 2.0
    assert rates[key] >= 0.0
