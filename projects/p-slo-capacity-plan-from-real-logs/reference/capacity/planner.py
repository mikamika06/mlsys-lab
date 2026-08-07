import json
import numpy as np


def parse_metrics_series(metrics_jsonl_path):
    """Parse jsonl metric lines into aggregated arrays."""
    timestamps = []
    gpu_util = []
    kv_cache_usage = []
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            timestamps.append(item["timestamp"])
            gpu_util.append(item["gpu_utilization"])
            kv_cache_usage.append(item["kv_cache_usage"])
    return {
        "timestamp": np.array(timestamps, dtype=np.float64),
        "gpu_utilization": np.array(gpu_util, dtype=np.float64),
        "kv_cache_usage": np.array(kv_cache_usage, dtype=np.float64),
    }


def parse_bench_results(bench_json_path):
    """Parse benchmark JSON containing load test runs."""
    with open(bench_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"]


def calculate_goodput(bench_data, max_p95_latency):
    """Extract runs that fulfill p95 SLO and calculate goodput in RPS and token throughput."""
    valid_runs = [r for r in bench_data if r["p95_latency"] <= max_p95_latency]
    if not valid_runs:
        return {"goodput_rps": 0.0, "goodput_tokens_per_sec": 0.0, "valid_runs": 0}

    best_run = max(valid_runs, key=lambda x: x["achieved_rps"])
    return {
        "goodput_rps": float(best_run["achieved_rps"]),
        "goodput_tokens_per_sec": float(best_run["output_tokens_per_sec"]),
        "valid_runs": len(valid_runs),
    }


def find_knee_capacity(bench_data, target_p95):
    """Find maximum offered RPS where p95 latency remains <= target_p95."""
    valid_rps = [r["offered_rps"] for r in bench_data if r["p95_latency"] <= target_p95]
    if not valid_rps:
        return 0.0
    return float(max(valid_rps))


def compute_required_replicas(target_rps, single_replica_capacity, headroom_factor=1.2):
    """Compute integer replicas required to sustain target RPS with safety headroom."""
    if single_replica_capacity <= 0:
        raise ValueError("Single replica capacity must be positive")
    needed_capacity = target_rps * headroom_factor
    replicas = int(np.ceil(needed_capacity / single_replica_capacity))
    return max(1, replicas)


def compute_cost_per_million_tokens(replica_count, hourly_cost_per_replica, rps, avg_output_tokens):
    """Compute cost in USD per 1M output tokens."""
    total_hourly_cost = replica_count * hourly_cost_per_replica
    tokens_per_sec = rps * avg_output_tokens
    tokens_per_hour = tokens_per_sec * 3600.0
    if tokens_per_hour <= 0:
        return 0.0
    cost_per_token = total_hourly_cost / tokens_per_hour
    return float(cost_per_token * 1e6)


def compute_prefix_cache_impact(target_rps, base_single_replica_capacity, hit_rate, speedup_factor, headroom_factor=1.2):
    """Compute new capacity and required replicas when prefix caching is enabled."""
    effective_capacity = base_single_replica_capacity * (1.0 + hit_rate * (speedup_factor - 1.0))
    new_replicas = compute_required_replicas(target_rps, effective_capacity, headroom_factor)
    return {
        "effective_single_capacity": float(effective_capacity),
        "required_replicas": new_replicas,
    }
