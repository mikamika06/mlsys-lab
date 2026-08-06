import numpy as np


def calculate_summary(traces):
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


def compute_latency_degradation_ratio(low_concurrency_summary, high_concurrency_summary):
    low_p90 = low_concurrency_summary.get("p90_ttft_ms", 0.0)
    high_p90 = high_concurrency_summary.get("p90_ttft_ms", 0.0)
    if low_p90 <= 0:
        return 0.0
    return float(high_p90 / low_p90)
