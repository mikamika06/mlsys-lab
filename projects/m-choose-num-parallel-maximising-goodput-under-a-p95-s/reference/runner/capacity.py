import numpy as np


def select_optimal_num_parallel(benchmark_results, p95_slo_ms):
    """
    Select best num_parallel that maximizes goodput while keeping P95 latency <= p95_slo_ms.
    """
    valid_results = []
    for res in benchmark_results:
        latencies = res["latencies_ms"]
        if not latencies:
            continue
        p95 = float(np.percentile(latencies, 95))
        duration_s = res.get("duration_s", 1.0)
        successful_under_slo = sum(1 for l in latencies if l <= p95_slo_ms)
        goodput = successful_under_slo / duration_s
        if p95 <= p95_slo_ms:
            valid_results.append((goodput, res["num_parallel"], p95))

    if valid_results:
        valid_results.sort(key=lambda x: (-x[0], x[1]))
        best = valid_results[0]
        return {
            "num_parallel": best[1],
            "max_goodput": best[0],
            "p95_latency_ms": best[2],
        }

    sorted_results = sorted(benchmark_results, key=lambda x: x["num_parallel"])
    best_res = sorted_results[0]
    latencies = best_res["latencies_ms"]
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    goodput = sum(1 for l in latencies if l <= p95_slo_ms) / best_res.get("duration_s", 1.0)
    return {
        "num_parallel": best_res["num_parallel"],
        "max_goodput": goodput,
        "p95_latency_ms": p95,
    }
