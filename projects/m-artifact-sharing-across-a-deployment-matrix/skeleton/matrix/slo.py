def calculate_required_instances(
    arrival_rate_rps: float,
    mean_service_time_ms: float,
    p99_service_time_ms: float,
    target_p99_slo_ms: float,
    max_utilization: float = 0.85,
) -> int:
    raise NotImplementedError
