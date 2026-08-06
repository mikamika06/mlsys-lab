import math

def compute_required_gpus(hourly_traffic_tok_per_sec: list[float], instance_throughput_tok_per_sec: float, gpus_per_instance: int, headroom_fraction: float = 0.30) -> dict:
    """Compute required instances and GPUs for traffic curve with headroom."""
    if instance_throughput_tok_per_sec <= 0:
        raise ValueError("Instance throughput must be positive")
    peak_traffic = max(hourly_traffic_tok_per_sec)
    required_capacity = peak_traffic * (1.0 + headroom_fraction)
    instances_needed = math.ceil(required_capacity / instance_throughput_tok_per_sec)
    total_gpus = instances_needed * gpus_per_instance
    return {
        "peak_traffic_tok_per_sec": peak_traffic,
        "target_capacity_tok_per_sec": required_capacity,
        "instances_needed": instances_needed,
        "total_gpus": total_gpus,
    }
