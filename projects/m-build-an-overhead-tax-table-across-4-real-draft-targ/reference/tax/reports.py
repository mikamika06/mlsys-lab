from tax.overhead import compute_pair_tax


def build_overhead_tax_table(pairs_data: list[dict], gammas: list[int]) -> list[dict]:
    """Build a consolidated overhead tax table across all pairs and gammas."""
    table = []
    for pair in pairs_data:
        for g in gammas:
            row = compute_pair_tax(pair, g)
            table.append(row)
    return table


def find_optimal_gamma(pair_data: dict, max_gamma: int) -> dict:
    """Find the speculation depth gamma that minimizes overhead tax."""
    best_tax = float("inf")
    best_res = None
    for g in range(1, max_gamma + 1):
        res = compute_pair_tax(pair_data, g)
        if res["overhead_tax"] < best_tax:
            best_tax = res["overhead_tax"]
            best_res = res
    return best_res
