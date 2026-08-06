def build_overhead_tax_table(pairs_data: list[dict], gammas: list[int]) -> list[dict]:
    """Build a consolidated overhead tax table across all pairs and gammas."""
    raise NotImplementedError


def find_optimal_gamma(pair_data: dict, max_gamma: int) -> dict:
    """Find the speculation depth gamma that minimizes overhead tax."""
    raise NotImplementedError
