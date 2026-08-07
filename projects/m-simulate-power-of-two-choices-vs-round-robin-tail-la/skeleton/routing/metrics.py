def get_per_replica_counts(num_replicas, requests, service_times, strategy="round_robin", seed=42):
    raise NotImplementedError

def compute_tail_latency(num_replicas, requests, service_times, strategy="round_robin", percentile=99, seed=42):
    raise NotImplementedError
