TIER_FACTORS = {"L0": 1.0, "L1": 0.85, "L2": 0.70}


def simulate_execution(
    stream: list[tuple[str, int]],
    base_freq_ghz: float,
    recovery_cycles: int,
) -> dict:
    """Simulate instruction stream execution under downclock recovery rules."""
    raise NotImplementedError
