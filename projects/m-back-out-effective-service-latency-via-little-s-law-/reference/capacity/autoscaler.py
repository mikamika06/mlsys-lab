import math


def evaluate_autoscaler(arrival_rate, service_rate_per_replica, replica_trace):
    """
    Computes minimum theoretical replicas and over-provisioning ratio against replica_trace.
    """
    min_replicas = math.ceil(arrival_rate / service_rate_per_replica)
    if not replica_trace:
        return {"min_replicas": min_replicas, "avg_actual_replicas": 0.0, "slack_ratio": 0.0}
        
    avg_actual = sum(replica_trace) / len(replica_trace)
    slack_ratio = (avg_actual - min_replicas) / min_replicas if min_replicas > 0 else 0.0
    return {
        "min_replicas": min_replicas,
        "avg_actual_replicas": avg_actual,
        "slack_ratio": slack_ratio,
    }
