import numpy as np


def generate_benchmark_data():
    return [
        {
            "num_parallel": 1,
            "latencies_ms": [80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
            "duration_s": 1.0,
        },
        {
            "num_parallel": 2,
            "latencies_ms": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "duration_s": 1.0,
        },
        {
            "num_parallel": 4,
            "latencies_ms": [120, 130, 140, 150, 160, 180, 220, 240, 260, 280],
            "duration_s": 1.0,
        },
        {
            "num_parallel": 8,
            "latencies_ms": [200, 220, 240, 260, 300, 350, 400, 450, 500, 600],
            "duration_s": 1.0,
        },
    ]


def select_optimal_num_parallel_oracle(benchmark_results, p95_slo_ms):
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


def generate_trace_data():
    return [
        {"id": "r1", "prefill_ms": 800.0, "wait_ms": 0.0, "status": 200},
        {"id": "r2", "prefill_ms": 20.0, "wait_ms": 350.0, "status": 200},
        {"id": "r3", "prefill_ms": 15.0, "wait_ms": 400.0, "status": 503, "dropped": True},
        {"id": "r4", "prefill_ms": 25.0, "wait_ms": 410.0, "status": 503, "dropped": True},
    ]
