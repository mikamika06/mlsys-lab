def compute_tps_multiplier(acceptance_rate: float) -> float:
    """Compute TPS multiplier from acceptance rate."""
    return float(1.0 + acceptance_rate * 0.8)
