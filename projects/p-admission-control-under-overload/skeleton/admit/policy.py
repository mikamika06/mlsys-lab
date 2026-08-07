def estimate_latency(queue_load: int, service_rate: float) -> float:
    raise NotImplementedError

def should_admit(estimated_latency: float, slo_target: float, priority: int) -> bool:
    raise NotImplementedError
