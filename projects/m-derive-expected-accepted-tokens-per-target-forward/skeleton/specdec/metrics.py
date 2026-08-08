def measure_acceptance_rates(accepted_counts: list[int], draft_max: int) -> list[float]:
    """Calculate conditional acceptance probabilities per draft position."""
    raise NotImplementedError


def expected_accepted_tokens(acceptance_rates: list[float]) -> float:
    """Calculate expected accepted draft tokens per target forward pass."""
    raise NotImplementedError


def expected_total_tokens(acceptance_rates: list[float]) -> float:
    """Calculate expected total tokens generated per target forward pass."""
    raise NotImplementedError
