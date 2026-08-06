def masked_sum(data: list[float], mask: list[bool]) -> float:
    """Return the sum of data values where mask is True.

    Uses an explicit loop to avoid the 0*inf NaN trap.
    """
    total = 0.0
    for d, m in zip(data, mask):
        if m:
            total += d
    return float(total)
