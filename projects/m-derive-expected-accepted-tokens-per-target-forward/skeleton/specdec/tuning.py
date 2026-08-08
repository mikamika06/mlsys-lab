def evaluate_draft_throughput(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> dict[int, float]:
    """Compute expected token generation throughput for draft lengths 1..N."""
    raise NotImplementedError


def select_optimal_draft_max(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> tuple[int, float]:
    """Select the draft_max length in 1..N that maximizes expected throughput."""
    raise NotImplementedError
