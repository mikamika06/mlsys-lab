import math
import numpy as np


def generate_trace_data(seed=42):
    rng = np.random.default_rng(seed)
    n_steps = 100
    timestamps = np.cumsum(rng.uniform(0.5, 1.5, size=n_steps))
    timestamps = np.insert(timestamps, 0, 0.0)
    
    active_requests = rng.integers(10, 50, size=len(timestamps))
    completed_requests = rng.integers(5, 20, size=len(timestamps)-1)
    
    trace_data = []
    for i in range(len(timestamps) - 1):
        trace_data.append({
            "timestamp_sec": float(timestamps[i]),
            "active_requests": int(active_requests[i]),
            "completed_requests": int(completed_requests[i])
        })
    trace_data.append({
        "timestamp_sec": float(timestamps[-1]),
        "active_requests": int(active_requests[-1]),
        "completed_requests": 0
    })
    return trace_data


def ref_compute_effective_latency(trace_data):
    total_l = 0.0
    total_time = 0.0
    total_completed = 0
    
    for i in range(len(trace_data) - 1):
        dt = trace_data[i + 1]["timestamp_sec"] - trace_data[i]["timestamp_sec"]
        if dt <= 0:
            continue
        total_time += dt
        total_l += trace_data[i]["active_requests"] * dt
        total_completed += trace_data[i]["completed_requests"]
        
    avg_l = total_l / total_time
    arrival_rate = total_completed / total_time
    return avg_l / arrival_rate


def generate_batch_profiles():
    return [
        {"batch_size": 1, "latency_ms": 12.0, "tokens_per_sec": 120.0},
        {"batch_size": 2, "latency_ms": 18.0, "tokens_per_sec": 280.0},
        {"batch_size": 4, "latency_ms": 28.0, "tokens_per_sec": 550.0},
        {"batch_size": 8, "latency_ms": 45.0, "tokens_per_sec": 950.0},
        {"batch_size": 16, "latency_ms": 85.0, "tokens_per_sec": 1400.0},
        {"batch_size": 32, "latency_ms": 160.0, "tokens_per_sec": 1800.0},
    ]


def ref_find_optimal_batch_size(profiles, sla_latency_ms, cost_per_node_hr):
    best_batch = None
    min_cost = float("inf")
    for p in profiles:
        if p["latency_ms"] <= sla_latency_ms:
            cost_per_sec = cost_per_node_hr / 3600.0
            cost_per_1k = (cost_per_sec / p["tokens_per_sec"]) * 1000.0
            if cost_per_1k < min_cost:
                min_cost = cost_per_1k
                best_batch = p["batch_size"]
    return {"optimal_batch_size": best_batch, "min_cost_per_1k_tokens": min_cost}


def generate_autoscaler_scenario():
    arrival_rate = 450.0
    service_rate = 50.0
    replica_trace = [12, 11, 13, 12, 14, 11, 12, 13]
    return arrival_rate, service_rate, replica_trace


def ref_evaluate_autoscaler(arrival_rate, service_rate_per_replica, replica_trace):
    min_replicas = math.ceil(arrival_rate / service_rate_per_replica)
    avg_actual = sum(replica_trace) / len(replica_trace)
    slack_ratio = (avg_actual - min_replicas) / min_replicas
    return {
        "min_replicas": min_replicas,
        "avg_actual_replicas": avg_actual,
        "slack_ratio": slack_ratio,
    }
