def compute_hardware_penalty(
    engine_sm: tuple, target_sm: tuple, base_latency_ms: float
) -> float:
    """Computes estimated hardware incompatibility execution penalty."""
    raise NotImplementedError
