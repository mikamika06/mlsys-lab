import numpy as np


def parse_histogram(metric_lines):
    buckets = {}
    sum_val = 0.0
    count_val = 0.0
    for line in metric_lines:
        if line.startswith("vllm_request_latency_seconds_bucket"):
            parts = line.split()
            if len(parts) >= 2:
                le = parts[0].split("le=\"")[1].split("\"")[0]
                val = float(parts[1])
                if le != "+Inf":
                    buckets[float(le)] = val
                else:
                    buckets[float("inf")] = val
        elif line.startswith("vllm_request_latency_seconds_sum"):
            sum_val = float(line.split()[1])
        elif line.startswith("vllm_request_latency_seconds_count"):
            count_val = float(line.split()[1])
    return {
        "buckets": buckets,
        "sum": sum_val,
        "count": count_val
    }
