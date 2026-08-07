def total_model_bytes(modules: list[dict], bit_assignment: dict[str, int]) -> float:
    """Compute the total model footprint in bytes given a bit assignment."""
    raise NotImplementedError


def uniform_allocation(
    modules: list[dict],
    exclude_patterns: list[str],
    target_bits: int,
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Produce a uniform bit allocation while keeping excluded modules at default FP bits."""
    raise NotImplementedError


def greedy_allocation(
    modules: list[dict],
    sensitivity_profile: dict[str, dict[int, float]],
    exclude_patterns: list[str],
    max_bytes: float,
    candidate_bits: list[int],
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Perform greedy bit allocation based on marginal error reduction per byte."""
    raise NotImplementedError


def optimal_allocation(
    modules: list[dict],
    sensitivity_profile: dict[str, dict[int, float]],
    exclude_patterns: list[str],
    max_bytes: float,
    candidate_bits: list[int],
    default_fp_bits: int = 16,
) -> dict[str, int]:
    """Find optimal bit allocation minimizing overall error under max_bytes using dynamic programming."""
    raise NotImplementedError


def construct_greedy_failure_case() -> tuple[list[dict], dict[str, dict[int, float]], float]:
    """Construct a synthetic problem instance where greedy allocation fails to find the optimum."""
    raise NotImplementedError
