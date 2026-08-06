import math


def calculate_required_instances(
    arrival_rate_rps: float,
    mean_service_time_ms: float,
    p99_service_time_ms: float,
    target_p99_slo_ms: float,
    max_utilization: float = 0.85,
) -> int:
    service_rate_per_instance = 1000.0 / mean_service_time_ms
    min_instances_capacity = math.ceil(arrival_rate_rps / (service_rate_per_instance * max_utilization))

    for instances in range(max(1, min_instances_capacity), 10000):
        total_capacity = instances * service_rate_per_instance
        rho = arrival_rate_rps / total_capacity
        if rho >= 1.0:
            continue
        queue_wait_ms = (rho / (1.0 - rho)) * mean_service_time_ms / instances
        estimated_p99_ms = p99_service_time_ms + 2.326 * queue_wait_ms
        if estimated_p99_ms <= target_p99_slo_ms and rho <= max_utilization:
            return instances
    return min_instances_capacity
