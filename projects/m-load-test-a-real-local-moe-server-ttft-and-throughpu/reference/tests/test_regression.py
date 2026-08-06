import sys

sys.path.insert(0, ".")
from moeload.benchmark import run_benchmark_session
from moeload.metrics import calculate_summary, compute_latency_degradation_ratio
from moeload.server import MoEServer


def test_ttft_is_positive_and_above_prefill():
    server = MoEServer(num_experts=8, active_experts=2, base_prefill_ms=10.0, gen_ms_per_tok=2.0)
    workload = [{"prompt_tokens": 100, "decode_tokens": 10}]
    traces = run_benchmark_session(server, workload, concurrency=1)
    summary = calculate_summary(traces)
    assert traces[0]["ttft_ms"] > 0.0, "TTFT must be strictly positive"
    assert summary["p50_ttft_ms"] >= 10.0, f"Expected P50 TTFT >= 10.0, got {summary['p50_ttft_ms']}"


def test_concurrency_degradation_ratio_positive():
    server = MoEServer(num_experts=8, active_experts=2, base_prefill_ms=10.0, gen_ms_per_tok=2.0)
    workload = [{"prompt_tokens": 100, "decode_tokens": 20}] * 5
    low_traces = run_benchmark_session(server, workload, concurrency=1)
    high_traces = run_benchmark_session(server, workload, concurrency=10)
    low_sum = calculate_summary(low_traces)
    high_sum = calculate_summary(high_traces)
    ratio = compute_latency_degradation_ratio(low_sum, high_sum)
    assert ratio > 1.0, f"Expected high concurrency to increase TTFT latency ratio > 1.0, got {ratio}"
