def welford_variance(data: list[float]) -> float:
    # WRONG IMPLEMENTATION: naive variance, susceptible to catastrophic cancellation
    n = len(data)
    if n < 1:
        return 0.0
    sum_x = sum(data)
    sum_x2 = sum(x*x for x in data)
    return (sum_x2 - (sum_x * sum_x) / n) / n
