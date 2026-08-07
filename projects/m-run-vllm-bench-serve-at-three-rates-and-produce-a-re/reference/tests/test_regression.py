"""Regression tests for benchmark harness."""

import sys

sys.path.insert(0, ".")
from bench_serve.bundle import create_result_bundle
from bench_serve.runner import run_multi_rate_bench

REQUESTS = [{"prompt_tokens": 64, "max_tokens": 32} for _ in range(20)]
RATES = [2.0, 5.0, 10.0]


def test_bundle_contains_all_requests():
    raw_data = run_multi_rate_bench(REQUESTS, RATES)
    bundle = create_result_bundle("test-model", raw_data)

    for rate in RATES:
        rate_str = str(rate)
        assert rate_str in bundle["rates"]
        rate_entry = bundle["rates"][rate_str]
        assert rate_entry["metrics"]["completed_requests"] == len(REQUESTS)
        assert len(rate_entry["raw_results"]) == len(REQUESTS)


def test_throughput_increases_or_stabilizes():
    raw_data = run_multi_rate_bench(REQUESTS, RATES)
    bundle = create_result_bundle("test-model", raw_data)

    tp_low = bundle["rates"][str(RATES[0])]["metrics"][
        "total_throughput_tok_s"
    ]
    tp_high = bundle["rates"][str(RATES[-1])]["metrics"][
        "total_throughput_tok_s"
    ]
    assert tp_high >= tp_low
