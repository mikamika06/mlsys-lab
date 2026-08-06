def zigzag_assignment(num_ranks: int) -> list[int]:
    total = 2 * num_ranks
    out = [0] * total
    for r in range(num_ranks):
        out[r] = r
        out[total - 1 - r] = r
    return out
