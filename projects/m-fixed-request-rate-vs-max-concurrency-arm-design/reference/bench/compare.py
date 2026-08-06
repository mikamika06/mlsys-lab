import numpy as np

def build_comparison_table(fixed_runs, concurrency_runs):
    fr_tp = [r["request_throughput"] for r in fixed_runs]
    mc_tp = [r["request_throughput"] for r in concurrency_runs]
    fr_p99 = [r["p99_latency_ms"] for r in fixed_runs]
    mc_p99 = [r["p99_latency_ms"] for r in concurrency_runs]

    mean_fr_tp = float(np.mean(fr_tp)) if fr_tp else 0.0
    mean_mc_tp = float(np.mean(mc_tp)) if mc_tp else 0.0

    return {
        "fixed_rate_mean_throughput": mean_fr_tp,
        "max_concurrency_mean_throughput": mean_mc_tp,
        "fixed_rate_mean_p99": float(np.mean(fr_p99)) if fr_p99 else 0.0,
        "max_concurrency_mean_p99": float(np.mean(mc_p99)) if mc_p99 else 0.0,
        "throughput_ratio": mean_mc_tp / mean_fr_tp if mean_fr_tp > 0 else 0.0
    }
