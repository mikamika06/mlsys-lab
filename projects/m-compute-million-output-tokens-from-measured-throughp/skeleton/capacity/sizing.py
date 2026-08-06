def compute_required_gpus(hourly_traffic_tok_per_sec: list[float], instance_throughput_tok_per_sec: float, gpus_per_instance: int, headroom_fraction: float = 0.30) -> dict:
    """Compute required instances and GPUs for traffic curve with headroom."""
    raise NotImplementedError
