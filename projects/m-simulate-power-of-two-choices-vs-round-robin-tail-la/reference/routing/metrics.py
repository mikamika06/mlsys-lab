import numpy as np
from routing.sim import simulate_round_robin, simulate_power_of_two

def get_per_replica_counts(num_replicas, requests, service_times, strategy="round_robin", seed=42):
    if strategy == "round_robin":
        counts, _ = simulate_round_robin(num_replicas, requests, service_times, seed=seed)
    else:
        counts, _ = simulate_power_of_two(num_replicas, requests, service_times, seed=seed)
    return counts

def compute_tail_latency(num_replicas, requests, service_times, strategy="round_robin", percentile=99, seed=42):
    if strategy == "round_robin":
        _, latencies = simulate_round_robin(num_replicas, requests, service_times, seed=seed)
    else:
        _, latencies = simulate_power_of_two(num_replicas, requests, service_times, seed=seed)
    return float(np.percentile(latencies, percentile))
