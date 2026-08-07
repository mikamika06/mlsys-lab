def check_memory_growth(allocations):
    peak = 0
    current = 0
    for a in allocations:
        current += a
        if current > peak:
            peak = current
    return {"peak": peak, "stable": current <= peak}
