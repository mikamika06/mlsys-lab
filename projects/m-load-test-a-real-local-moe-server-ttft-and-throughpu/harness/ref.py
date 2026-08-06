import numpy as np

WORKLOADS = [
    [{"prompt_tokens": 100, "decode_tokens": 20}],
    [
        {"prompt_tokens": 50, "decode_tokens": 10},
        {"prompt_tokens": 200, "decode_tokens": 50},
        {"prompt_tokens": 128, "decode_tokens": 32},
    ],
    [{"prompt_tokens": 500, "decode_tokens": 100}] * 10,
]


class RefMoEServer:

    def __init__(self, num_experts=8, active_experts=2, base_prefill_ms=10.0, gen_ms_per_tok=2.0):
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.base_prefill_ms = base_prefill_ms
        self.gen_ms_per_tok = gen_ms_per_tok

    def process_request(self, prompt_tokens, decode_tokens, concurrency=1):
        expert_factor = 1.0 + (self.active_experts / float(self.num_experts))
        contention_factor = 1.0 + 0.15 * (concurrency - 1)

        prefill_time = self.base_prefill_ms * (prompt_tokens / 100.0) * expert_factor * contention_factor
        gen_time_per_token = self.gen_ms_per_tok * contention_factor

        ttft = prefill_time + gen_time_per_token
        inter_token_latencies = [gen_time_per_token] * max(0, decode_tokens - 1)
        total_time = ttft + sum(inter_token_latencies)

        return {
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "ttft_ms": ttft,
            "inter_token_latencies_ms": inter_token_latencies,
            "total_time_ms": total_time,
            "total_tokens": prompt_tokens + decode_tokens,
        }


def ref_run_benchmark_session(server, workload_requests, concurrency):
    traces = []
    for req in workload_requests:
        prompt_toks = req["prompt_tokens"]
        decode_toks = req["decode_tokens"]
        trace = server.process_request(prompt_toks, decode_toks, concurrency=concurrency)
        traces.append(trace)
    return traces


def ref_calculate_summary(traces):
    if not traces:
        return {
            "total_requests": 0,
            "throughput_tok_per_sec": 0.0,
            "p50_ttft_ms": 0.0,
            "p90_ttft_ms": 0.0,
            "p99_ttft_ms": 0.0,
            "mean_ttft_ms": 0.0,
        }

    ttfts = np.array([t["ttft_ms"] for t in traces], dtype=np.float64)
    total_tokens = sum(t["total_tokens"] for t in traces)
    max_total_time_s = max(t["total_time_ms"] for t in traces) / 1000.0

    throughput = total_tokens / max_total_time_s if max_total_time_s > 0 else 0.0

    return {
        "total_requests": len(traces),
        "throughput_tok_per_sec": float(throughput),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p90_ttft_ms": float(np.percentile(ttfts, 90)),
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "mean_ttft_ms": float(np.mean(ttfts)),
    }


def ref_compute_latency_degradation_ratio(low_concurrency_summary, high_concurrency_summary):
    low_p90 = low_concurrency_summary.get("p90_ttft_ms", 0.0)
    high_p90 = high_concurrency_summary.get("p90_ttft_ms", 0.0)
    if low_p90 <= 0:
        return 0.0
    return float(high_p90 / low_p90)
