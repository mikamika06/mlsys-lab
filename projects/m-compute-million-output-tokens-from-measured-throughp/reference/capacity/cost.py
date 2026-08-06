def compute_cost_per_million_tokens(throughput_tok_per_sec: float, hourly_instance_price: float) -> float:
    """Compute cost per million output tokens."""
    if throughput_tok_per_sec <= 0:
        raise ValueError("Throughput must be positive")
    tokens_per_hour = throughput_tok_per_sec * 3600.0
    return (hourly_instance_price / tokens_per_hour) * 1_000_000.0
