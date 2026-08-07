import json
import numpy as np


def generate_fixtures(workdir):
    metrics_path = f"{workdir}/metrics.jsonl"
    bench_path = f"{workdir}/bench_results.json"

    metrics_data = [
        {"timestamp": 1700000000.0 + i, "gpu_utilization": 0.4 + 0.01 * (i % 50), "kv_cache_usage": 0.2 + 0.005 * i}
        for i in range(100)
    ]
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in metrics_data:
            f.write(json.dumps(m) + "\n")

    runs = [
        {"offered_rps": 10, "achieved_rps": 10.0, "p95_latency": 0.8, "output_tokens_per_sec": 1000.0},
        {"offered_rps": 20, "achieved_rps": 19.8, "p95_latency": 1.4, "output_tokens_per_sec": 1980.0},
        {"offered_rps": 30, "achieved_rps": 28.5, "p95_latency": 2.2, "output_tokens_per_sec": 2850.0},
        {"offered_rps": 40, "achieved_rps": 32.0, "p95_latency": 3.8, "output_tokens_per_sec": 3200.0},
        {"offered_rps": 50, "achieved_rps": 33.1, "p95_latency": 6.5, "output_tokens_per_sec": 3310.0},
    ]
    bench_data = {"runs": runs}
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f)

    return metrics_path, bench_path


def oracle_parse_metrics(path):
    timestamps, gpu, kv = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                timestamps.append(item["timestamp"])
                gpu.append(item["gpu_utilization"])
                kv.append(item["kv_cache_usage"])
    return {
        "timestamp": np.array(timestamps, dtype=np.float64),
        "gpu_utilization": np.array(gpu, dtype=np.float64),
        "kv_cache_usage": np.array(kv, dtype=np.float64),
    }


def oracle_parse_bench(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["runs"]
