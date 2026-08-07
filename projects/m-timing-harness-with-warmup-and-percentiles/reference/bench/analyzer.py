import numpy as np


def find_benchmark_bug(run_data):
    timer_type = run_data.get("timer_type", "perf_counter")
    if timer_type != "perf_counter":
        return "LOW_PRECISION_TIMER"

    warmup_count = run_data.get("warmup_count", 0)
    raw_latencies = run_data.get("raw_latencies", [])

    if len(raw_latencies) <= warmup_count:
        return "INSUFFICIENT_SAMPLES"

    measured = raw_latencies[warmup_count:]
    reported_metrics = run_data.get("reported_metrics", {})

    if warmup_count > 0 and len(raw_latencies) > warmup_count:
        first_warmup = raw_latencies[0]
        rest_mean = float(np.mean(measured))
        if first_warmup > rest_mean * 2.0:
            if "p50" in reported_metrics:
                all_p50 = float(np.percentile(raw_latencies, 50))
                rep_p50 = reported_metrics["p50"]
                if abs(rep_p50 - all_p50) < abs(rep_p50 - float(np.percentile(measured, 50))) or abs(rep_p50 - first_warmup) < 1e-6:
                    return "WARMUP_INCLUDED_IN_METRICS"

    sorted_m = sorted(measured)
    n = len(sorted_m)
    if "p99" in reported_metrics and n > 1:
        rep_p99 = reported_metrics["p99"]
        rank = 0.99 * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        w = rank - lo
        expected_p99 = sorted_m[lo] * (1.0 - w) + sorted_m[hi] * w
        if abs(rep_p99 - expected_p99) > 1e-5:
            return "INCORRECT_PERCENTILE_CALCULATION"

    if len(measured) < 30:
        return "TOO_FEW_RUNS_FOR_STABILITY"

    return "NONE"
