def estimate_latency(queue_load: int, service_rate: float) -> float:
    if service_rate <= 0:
        return float('inf')
    return float(queue_load) / service_rate

def should_admit(estimated_latency: float, slo_target: float, priority: int) -> bool:
    adjusted_slo = slo_target * (1.0 + 0.1 * float(priority))
    return estimated_latency <= adjusted_slo
